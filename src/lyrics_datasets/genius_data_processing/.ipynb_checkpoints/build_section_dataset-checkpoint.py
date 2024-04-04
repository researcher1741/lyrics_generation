from collections import defaultdict
from typing import Dict
import jsonlines
from prometheus_client import Counter
from tqdm import tqdm
import itertools
from argparse import ArgumentParser
from phonemizer.separator import Separator
import math
from phonemizer import phonemize
from phonemizer.punctuation import Punctuation
from lyrics_generation_utils.constants import BLOCK_END, SENTENCE_END, TAG_END
from collections import Counter
import lyrics_generation_utils.utils as lg_utils
import re
from multiprocessing import Pool
from multiprocessing import Queue, Manager, Process
import subprocess
from lyrics_datasets.phonetics import near_rhyme, perfect_rhyme

def normalise_tag(tag):
    '''
    This method normalise tags such as [Verse X], [Chorus Y], [Pre-Chorus Z] etc.
    As for now, it removes info about who is singing the part, e.g., [Verse 1: Celine Dion] [Chorus: Eminem] 
    become [Verse 1] [Chorus] in order to make the model generalise better across songs, but this may not be
    the best possible choice.
    '''
    tag = tag.split(':')[0]
    if not tag.endswith(']'): 
        tag = tag + ']'
    return tag

def prepare_blocks(blocks, eblocks, tag, schema):
    if len(schema) == len(set(schema)) or len(schema) > 24:
        return [], [], []
    blocks = [x + [SENTENCE_END] for x in  blocks]
    eblocks = [x + [SENTENCE_END] for x in eblocks]
    new_sentences = list()
    repeated_letters = set(map(lambda x: x[0], filter(lambda x: x[1] > 1, Counter(schema).most_common(len(schema)))))
    new_l = 'A'
    letter_mapping = dict()
    new_schema = []
    for l in sorted(list(repeated_letters)):
        letter_mapping[l] = new_l
        new_l = chr(ord(new_l) + 1)
    for sentence, letter in zip(blocks, schema):
        if len(sentence) == 1:
            continue
        idx = -2
        if sentence[idx].endswith(')') :
            while abs(idx) <= len(sentence) and not sentence[idx].startswith('('):
                idx -= 1
            if abs(idx) >= len(sentence):
                idx = -1
            idx -= 1

        if letter not in repeated_letters:
            letter = 'Z'
        else:
            letter = letter_mapping[letter]
        sentence = sentence[:idx] + ['RHYME_' + letter] + sentence[idx:]
        new_schema.append(letter)
        new_sentences.append(sentence)
    # if len(tag) > 0:
    #     blocks = [[tag + TAG_END]] + blocks
    #     eblocks = [[TAG_END]] + eblocks
    return new_sentences, eblocks, new_schema
    
def build_single_process(phonemised_genius_path, out_path):
    skipped = 0
    num_lines = int(subprocess.check_output(['wc', '-l', phonemised_genius_path]).decode().split(' ')[0])
    with jsonlines.open(phonemised_genius_path) as lines, jsonlines.open(out_path, 'w') as writer:
        for song in tqdm(lines, desc='building dataset', total=num_lines):
            title = song['title']
            artist = song['artist']
            genre = song['genre']
            emotions = song['emotion_tags']
            topics = song['topics']
            lyrics = song['tokenized_lyrics']
            lyrics_blocks = song['lyrics']
            blocks = [x.strip() for x in lyrics_blocks.split(BLOCK_END)]
            tags = [x.split('\n')[0] for x in blocks]
            tags = [t if t.startswith('[') else '' for t in tags if t != '']
            espeak_lyrics = song['espeak_tokenized_lyrics']
            rhyming_schema = song['schema']
            num_syllables = song.get("num_syllables",[])
            if len(rhyming_schema) == 0 or len(lyrics) == 0:
                skipped += 1
                continue
            prompt = {'title':title, 
                'artist':artist, 
                'genre':genre,
                'emotions': ','.join(emotions),
                'topics':','.join(topics)
                }
            for i, (block_l, block_e, schema, n_syl, tag) in enumerate(zip(lyrics, espeak_lyrics, rhyming_schema, num_syllables, tags)):
                if len(set(schema)) == len(schema): # no rhymes
                    skipped += 1
                    continue
                if len(schema) < 2:
                    skipped += 1
                    continue
                block_l, block_e, schema = prepare_blocks(block_l, block_e, tag, schema)
                if len(block_l) == 0:
                    continue
                aux = dict(prompt)
                aux.update({
                    'lyrics': block_l, 
                    # 'espeak_lyrics': block_e, 
                    'rhyming_schema': schema, 
                    'num_syllables': n_syl,
                    'prompt_lyrics': None,
                    'prompt_espeak_lyrics': None,
                    'prompt_rhyming_schema': None,
                    'prompt_num_syllables': None
                    })
                writer.write(aux)
                if i < len(lyrics) - 1:
                    aux = dict(prompt)
                    l, e, rs = prepare_blocks(lyrics[i+1], espeak_lyrics[i+1], tags[i+1], rhyming_schema[i+1])
                    if len(l) == 0:
                        continue
                    if len(set(rs)) < len(rs) and len(rs) >= 2:
                        aux.update({
                                    'lyrics': l, 
                                    # 'espeak_lyrics': e, 
                                    'rhyming_schema': rs, 
                                    'num_syllables':num_syllables[i+1],
                                    'prompt_lyrics': block_l,
                                    # 'prompt_espeak_lyrics': block_e,
                                    'prompt_rhyming_schema': schema,
                                    'prompt_num_syllables': n_syl
                                    })
                        
                        writer.write(aux)
    print('skipped songs:', skipped)

   

def build_genius_dataset(genius_path, out_path):
    """
    From a list of songs to a list of 
        - <prompt> + <verse> 
        - <prompt> + <chorus> 
        - <prompt> + <verse> + <chorus> 
        - <prompt> + <chorus> + <verse>
        - <prompt> + <verse> + <verse>
    Each item is composed as follows:
    {
        "prompt": {"artist": "Bruce Springsteen", "title": "The River", ..., "block_tag":"VERSE1", "block_text":"bla bla bla"},
        "block_tag":"CHORUS",
        "block_text":"bla bla bla"
    }
    where "block_tag" and "block_text" in "prompt" are optional.
    """
    build_single_process(genius_path, out_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--phonemized_genius_path', default='data/genius/phonemised_dataset/phonemised_dataset.jsonl')
    parser.add_argument('--out_path', default='data/genius/phonemised_dataset/section_dataset.jsonl')
    args = parser.parse_args()
    build_genius_dataset(args.phonemized_genius_path, args.out_path)

