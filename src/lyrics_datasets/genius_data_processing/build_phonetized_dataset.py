from collections import defaultdict
from typing import Dict
import jsonlines
from tqdm import tqdm
import itertools
from argparse import ArgumentParser
from phonemizer.separator import Separator
from phonemizer import phonemize
from phonemizer.punctuation import Punctuation
import lyrics_generation_utils.utils as lg_utils
import re
from multiprocessing import Pool
from multiprocessing import Queue, Manager, Process
from lyrics_datasets.phonetics import near_rhyme, perfect_rhyme
from lyrics_generation_utils.utils import PhonemeFactory, count_syllables, infer_rhyming_schema
from lyrics_generation_utils.constants import BLOCK_END, SENTENCE_END

def normalise_tag(tag):
    '''
    This method normalise tags such as [Verse X], [Chorus Y], [Pre-Chorus Z] etc.
    As for now, it removes info about who is singing the part, e.g., [Verse 1: Celine Dion] [Chorus: Eminem] 
    become [Verse 1] [Chorus] in order to make the model generalise better across songs, but this may not be
    the desired behaviour.
    '''
    tag = tag.split(':')[0]
    if not tag.endswith(']'): 
        tag = tag + ']'
    return tag

def split_lyrics(lyrics):
    # blocks = lyrics.split('\n[')
    blocks = [x.strip() for x in lyrics.split(BLOCK_END)]
    if len(blocks) == 1:
        blocks = lyrics.split('\n\n')
    aux = []
    for b in blocks:
        idx = b.find(']')
        if idx >= 0:
            tag = b[0:idx+1].strip()
            if not tag.startswith('['):
                tag = '[' + tag
            tag = normalise_tag(tag)
        else:
            tag = ''
            idx = -1
        block = b[idx+1:].strip()
        if len(block) > 0:
            aux.append((tag, block))
    return aux


def get_phonetics(blocks, phoneme_factory):
    texts = [x[1].replace(SENTENCE_END, '') for x in blocks]

    text_words = [w for line in texts for w in line.split()]

    espeak, festival = lg_utils.get_phonetics(text_words, phoneme_factory, espeak_separator=Separator(phone='.', syllable='', word=''), 
                                            festival_seapartor=Separator(phone='.', syllable='|', word=''))
    text_blocks = []
    espeak_bloks = []
    festival_blocks = []
    i = 0
    for l in texts:
        block_lines = l.split('\n')
        block_words = []
        block_espeak = []
        block_festival = []
        for line in block_lines:
            num_line_words = len(line.split())
            block_words.append(line.split())
            if espeak is not None:
                block_espeak.append(espeak[i:i+num_line_words])
                block_festival.append(festival[i:i+num_line_words])
            i+=num_line_words
        
        text_blocks.append(block_words)
        if espeak is not None:
            espeak_bloks.append(block_espeak)
            festival_blocks.append(block_festival)
    if espeak is not None:
        return text_blocks, espeak_bloks, festival_blocks
    return text_blocks, None, None

def add_syllables(song):
    festival = song['festival_tokenized_lyrics']
    if festival is not None:
        syllables = list()
        for block in festival:
            num_syllables = count_syllables([' '.join(l) for l in block])
            syllables.append(num_syllables)
        song['num_syllables'] = syllables
    return song

def do_rhyme(w1:Dict, w2:Dict, use_similar_words=False):
    if perfect_rhyme(w1, w2):
        return True
    return use_similar_words and near_rhyme(w1, w2)

def add_rhyming_schema(item, sentence_window = 4, use_similar_to=False):
    text = item['block_text']
    schema = infer_rhyming_schema(text, item['block_phonetics_espeak'], item['block_phonetics_festival'], 
    sentence_window, use_similar_to)
    item['rhyming_schema'] = schema
    

def process_song(song, phoneme_factory):
    lyrics = song['lyrics'].replace('\u205f', ' ').replace('\u200b', ' ')
    lyrics_blocks = split_lyrics(lyrics) 
    text_blocks, espeak_phonemes_blocks, festival_phonemes_blocks = get_phonetics(lyrics_blocks, phoneme_factory)
    assert espeak_phonemes_blocks is None or (len(text_blocks) == len(espeak_phonemes_blocks) == len(festival_phonemes_blocks))
    song['tokenized_lyrics'] = text_blocks
    song['espeak_tokenized_lyrics'] = espeak_phonemes_blocks
    song['festival_tokenized_lyrics'] = festival_phonemes_blocks
    return song

def song_processor(queue:Queue, out_queue:Queue, sentence_window=4):
    phoneme_factory = PhonemeFactory(['espeak', 'festival'], language='en-us',
                    preserve_punctuation=True,
                    punctuation_marks=Punctuation.default_marks()+'()[]{}‘-’\'\n',
                    with_stress=True)
    while True:
        try:
            song = queue.get()
            if song is None:
                break
            song = process_song(song, phoneme_factory)
            song = add_syllables(song)
            schemas = []
            if song['espeak_tokenized_lyrics'] is not None:
                for block, e_block, f_block in zip(song['tokenized_lyrics'], song['espeak_tokenized_lyrics'], song['festival_tokenized_lyrics']):
                    text = '\n'.join([' '.join(sentence) for sentence in block])
                    e_text = '\n'.join([' '.join(sentence) for sentence in e_block])
                    f_text = '\n'.join([' '.join(sentence) for sentence in f_block])
                    schema = infer_rhyming_schema(text, e_text, f_text, sentence_window, True) 
                    schemas.append(schema)
            song['schema'] = schemas
            out_queue.put(song)
        except RuntimeError as e:
            print(e)
            continue
    out_queue.put(None)

def song_writer_processor(queue, outfile, total_songs):
    with jsonlines.open(outfile, 'w') as writer:
        bar = tqdm(desc='examples writer', total=total_songs)
        while True:
            song = queue.get()
            if song is None:
                return
            bar.update()
            writer.write(song)


def build_parallel(genius_path, out_path, num_processes = 15, sentence_window=4):
    with jsonlines.open(genius_path) as lines:
        print('loading songs')
        songs = list(lines)
    songs_queue = Manager().Queue()
    examples_queue = Manager().Queue()
    processors = list()
    for _ in range(num_processes):
        p = Process(target=song_processor, args=[songs_queue, examples_queue, sentence_window])
        processors.append(p)
        p.start()
    writer_process = Process(target=song_writer_processor, args=[examples_queue, out_path, len(songs)])
    writer_process.start()
    for song in songs:
        songs_queue.put(song)
    for _ in range(num_processes):
        songs_queue.put(None)
    writer_process.join()
    for p in processors:
        p.join()

def build_single_process(genius_path, out_path, sentence_window=4):
    phoneme_factory = PhonemeFactory(['espeak', 'festival'], language='en-us',
                    preserve_punctuation=True,
                    punctuation_marks=Punctuation.default_marks()+'()[]{}‘-’\'\n',
                    with_stress=True)
        
    with jsonlines.open(genius_path) as lines, jsonlines.open(out_path, 'w') as writer:
        songs = list(lines)
        i = 0
        for song in tqdm(songs, desc='building blocks'):
            i+=1
            song = process_song(song, phoneme_factory)
            song = add_syllables(song)
            schemas = []
            if song['espeak_tokenized_lyrics'] is not None:
                for block, e_block, f_block in zip(song['tokenized_lyrics'],song['espeak_tokenized_lyrics'], song['festival_tokenized_lyrics']):
                    text = '\n'.join([' '.join(sentence) for sentence in block])
                    e_text = '\n'.join([' '.join(sentence) for sentence in e_block])
                    f_text = '\n'.join([' '.join(sentence) for sentence in f_block])
                    schema = infer_rhyming_schema(text, e_text, f_text, sentence_window, True) 
                    schemas.append(schema)
                song['schema'] = schemas

            writer.write(song)

def build_genius_dataset(genius_path, out_path, sentence_window):
    """
    Add phonetics information to a song, i.e., lyrics phonemes, syllable counting for each line
    and tries to infer the rhyming schema for each block (chorus, verse, etc.) in a song.

    It also divide lyrics in blocks, blocks in sentences, and sentences in words. Pronouciation, in this way, 
    is aligned word-to-word and can be alter be easily tokenized and sub-tokens associated to sub-phonemes.

    The method adds the following fields to the song dictionary:

    'tokenized_lyrics' - the lyrics split in blocks, sentences, and words
    'espeak_tokenized_lyrics' - the phonemes computed with espeak, good for computing rhymes, aligned to tokenized_lyrics
    'festival_tokenized_lyrics' - the phonemes computed with festival, good for syllable counting, aligned to tokenized_lyrics
    'schema' - the rhyming schema for each block
    'num_syllables' - number of syllables for each sentence in each block
    """

    build_parallel(genius_path, out_path, sentence_window)
    # build_single_process(genius_path, out_path, sentence_window)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--genius_path', default='data/genius/genius+wasabi-metadata.jsonl')
    parser.add_argument('--out_path', default='data/genius/section_dataset/phonemised_dataset.jsonl')
    parser.add_argument('--sentence_window', default=4)
    args = parser.parse_args()
    build_genius_dataset(args.genius_path, args.out_path, args.sentence_window)

