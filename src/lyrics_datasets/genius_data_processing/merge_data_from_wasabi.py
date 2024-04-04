import jsonlines
from argparse import ArgumentParser
from tqdm import tqdm
        
import spacy
import spacy_fastlang
import unicodedata, re, itertools, sys

from lyrics_generation_utils.utils import BLOCK_END, SENTENCE_END

def build_regex_for_control_char():
    all_chars = (chr(i) for i in range(sys.maxunicode))
    categories = {'Cc'}
    control_chars = ''.join(c for c in all_chars if unicodedata.category(c) in categories)
    # or equivalently and much more efficiently
    control_chars = ''.join(map(chr, itertools.chain(range(0x00,0x20), range(0x7f,0xa0))))

    control_char_re = re.compile('[%s]' % re.escape(control_chars))
    return control_char_re

CONTROL_CHAR_REGEX=build_regex_for_control_char()

def remove_control_chars(s:str):
    s = re.sub(r'\n\n\n+', '\n\n', s)
    s = s.replace('\n\n', '<NEWLINE><NEWLINE>')
    s = s.replace('\n', '<NEWLINE>')
    s = CONTROL_CHAR_REGEX.sub('', s)
    s = s.replace('\u200b', '')
    s = s.replace('<NEWLINE><NEWLINE>', '\n\n')
    s = s.replace('<NEWLINE>', '\n')
    return s

def divide_blocks(lyrics):
    blocks = lyrics.split('\n[')
    if len(blocks) == 1:
        blocks = lyrics.split('\n\n')
    else:
        for i in range(1, len(blocks)):
            blocks[i] = '[' + blocks[i]
    return [b + BLOCK_END if b[-1] == '\n' else b + f'\n{BLOCK_END}' for b in blocks]

def merge_with_wasabi_data(genius_path, wasabi_path, out_path, max_length):
    with jsonlines.open(wasabi_path) as lines:
        wasabi_songs = dict()
        for s in tqdm(lines, desc='reading wasabi'):
            title = remove_control_chars(s['title'].lower().replace('_', ' '))
            artist = remove_control_chars(s['artist'].lower().replace('_', ' '))
            wasabi_songs[(title, artist)] = s
    tot = 0
    merged = 0
    nlp = spacy.load("en_core_web_sm", exclude=['ner', 'tok2vec', 'lemmatizer', 'attribute_ruler', 'senter', 'parser', 'tagger'])
    nlp.add_pipe("language_detector")

    with jsonlines.open(out_path, 'w') as writer, jsonlines.open(genius_path) as lines:
        for song in tqdm(lines, desc='merging data'):
            lyrics = song['lyrics']
            if lyrics.startswith(song['title'] + ' Lyrics'):
                lyrics = lyrics[len(song['title'] + ' Lyrics'):]
            lyrics = lyrics.replace('\u2005', ' ').replace('\u205f', ' ').strip()
            if lyrics.endswith('Embed'):
                lyrics = lyrics[:-len('embed') + 1]
            lyrics = remove_control_chars(lyrics)
            if len(lyrics.split()) > max_length:
                continue
            doc = nlp(lyrics)
            if doc._.language != 'en':
                continue
            if len(lyrics.strip().split()) < 50:
                continue
            title = remove_control_chars(song['title'].lower().replace('_', ' '))
            artist = remove_control_chars(song['artist'].lower().replace('_', ' '))
            wasabi_song = wasabi_songs.get((title, artist))
            song['source'] = 'genius'
            lyrics = lyrics.replace('’', "'").replace('‘', "'")
            lyrics = '\n'.join([l + f' {SENTENCE_END}'  if len(l.strip()) > 0 and not l.startswith('[') and not l.endswith(']') else l for l in lyrics.split('\n')])
            lyrics = '\n\n'.join(divide_blocks(lyrics))
            
            song['lyrics'] = lyrics

            if wasabi_song is not None:
                if 'lyrics' in wasabi_song:
                    del wasabi_song['lyrics']
                if 'title' in wasabi_song:
                    del wasabi_song['title']
                if 'artist' in wasabi_song:
                    del wasabi_song['artist']

                song.update({k:wasabi_song[k] for k in ['lang', 'genre', 'emotion_tags', 'topics']})
                song['source'] += '+wasabi'
                merged += 1
            else:
                song.update({'lang': 'English', 
                'genre': '', 
                'emotion_tags': [], 
                'topics': []})
            song['title'] = remove_control_chars(song['title'])
            song['artist'] = remove_control_chars(song['artist'])
            writer.write(song)
            tot += 1
    print(f'Merged {merged} songs, {merged/tot * 100:.2f}% of the total.')
                
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--genius_path', default='data/genius/genius_lyrics.jsonl')
    parser.add_argument('--wasabi_path', default='data/wasabi/songs.jsonl')
    parser.add_argument('--out_path', default='data/genius/genius+wasabi-metadata.jsonl')
    ## this magic number derives from analysis done in jupyter and basically it is like
    # removing the top 1000 longest lyrics (yeet without needing to keep everything in memory and sorting them out)
    parser.add_argument('--max_length', default=1084) 

    args = parser.parse_args()
    merge_with_wasabi_data(args.genius_path, args.wasabi_path, args.out_path, args.max_length)
    # with jsonlines.open('data/genius_lyrics_333_666_wasabi.jsonl') as lines:
    #     for line in lines:
    #         if line['title'] == '1 4 U':
    #             print()