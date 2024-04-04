from argparse import ArgumentParser
from copy import copy
from typing import List
from src.lyrics_datasets.phonetics import Phonetic, near_rhyme
# from src.lyrics_generation_utils.constants import SENTENCE_END
from src.lyrics_generation_utils.constants import SENTENCE_END
import jsonlines
from tqdm import tqdm
from phonemizer.backend import BACKENDS
import re

language2code = {'italian': 'it',
                 'german': 'de',
                 'french': 'fr-fr',
                 'croatian': 'hr',
                 'danish': 'da',
                 'dutch': 'nl',
                 'finnish': 'fi',
                 'hungarian': 'hu',
                 'lithuanian': 'lt',
                 'norwegian': 'nb',
                 'polish': 'pl',
                 'portuguese': 'pt',
                 'slovak': 'sk',
                 'spanish': 'es',
                 'swedish': 'sv',
                 'turkish': 'tr',
                 'english': 'en-us',
                 'chinese': 'cmn'
                 }
JAPANESE_CHAR_REGEX = '[ぁ-んァ-ン！：／ァ-ン]'
KOREAN_CHAR_REGEX = '[\u3131-\uD79D]'


def build_artificial_blocks(lyrics, window_size=6) -> List[List[str]]:
    sentences = lyrics.split('\n')
    blocks = []
    for i in range(len(sentences), window_size):
        sentences = [sentence.strip() for sentence in sentences[i:i + window_size] if len(sentence.strip()) > 0]
        if len(sentences) > 2:
            blocks.append(sentences)
    return blocks


def split_blocks(blocks) -> List[List[str]]:
    aux = list()
    for block in blocks:
        sentences = [s.strip() for s in block.split('\n') if len(s.strip()) > 0]
        if len(sentences) > 2:
            aux.append(sentences)
    return aux


def tokenize(sentence, language):
    if sentence.endswith(')'):
        if '(' in sentence:
            sentence = sentence[:sentence.rindex('(')]
    return sentence.split()


def build_rhyming_schema(sentences, language, phonemizer):
    last_words = []
    new_sentences = []
    for sentence in sentences:
        if sentence == 'RIT':
            continue
        tokens = tokenize(sentence, language)
        if len(tokens) == 0:
            continue
        if 'RIT' in ' '.join(tokens):
            continue
        last_words.append(tokens[-1])
        new_sentences.append(tokens + ['<sentence_end>'])
    schema = [None for _ in range(len(last_words))]
    letter = 0
    offset = ord('A')
    for i in range(len(last_words)):
        if schema[i] is not None:
            continue
        a = last_words[i]
        a_phonetic = Phonetic(a, phonemizer=phonemizer)
        schema[i] = chr(letter + offset)
        for j in range(i + 1, len(last_words), 1):
            b = last_words[j]
            b_phonetic = Phonetic(b, phonemizer=phonemizer)
            if near_rhyme(a_phonetic, b_phonetic, language):
                schema[j] = chr(letter + offset)
        letter += 1
    return schema, new_sentences


def build_rhyming_schemas(blocks, language, phonemizer):
    schemas = list()
    new_blocks = list()
    for block in blocks:
        schema, block = build_rhyming_schema(block, language, phonemizer)
        schemas.append(schema)
        new_blocks.append(block)
    return schemas, new_blocks


def blockify_song(song, language, phonemizer):
    lyrics = song['lyrics']
    if language != 'japanese' and re.findall(JAPANESE_CHAR_REGEX, lyrics):
        return []
    if language != 'korean' and re.findall(KOREAN_CHAR_REGEX, lyrics):
        return []
    blocks = lyrics.split('\n\n')
    if len(blocks) == 1:
        blocks = build_artificial_blocks(lyrics)
    else:
        blocks: List[List[str]] = split_blocks(blocks)
    rhyming_schemas, blocks = build_rhyming_schemas(blocks, language, phonemizer)
    items = list()
    for i, (block, rhyming_schema) in enumerate(zip(blocks, rhyming_schemas)):
        str_block = SENTENCE_END.join([' '.join(sentence) for sentence in block])
        if len(str_block) == 0 or len(block) < 3:
            continue
        item = {'lyrics': block, 'schema': rhyming_schema, 'prompt_lyrics': None}
        item.update({k: song[k] for k in ['title', 'artist', 'emotion_tags', 'genre', 'topics']})
        items.append(item)
        if i > 0:
            item_copy = copy(item)
            prompt_lyrics = items[-1]['lyrics']
            item_copy['prompt_lyrics'] = prompt_lyrics
            items.append(item_copy)
    return items


def blockify_dataset(language, path, outpath, phonemizer):
    title_and_artist = set()
    with jsonlines.open(path) as lines, jsonlines.open(outpath, 'w') as writer:
        for song in tqdm(lines, 'blockifying ' + language):
            key = song['title'].lower().replace('_', ' ').replace(' ', '') + song['artist'].lower().replace('_',
                                                                                                            ' ').replace(
                ' ', '')
            if key in title_and_artist:
                continue
            title_and_artist.add(key)
            items = blockify_song(song, language, phonemizer)
            for item in items:
                writer.write(item)
    print('Done')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', default='data/wasabi/songs_by_language/italian.jsonl')
    parser.add_argument('--outpath', default='data/wasabi/songs_by_language/italian_section_dataset.jsonl')
    parser.add_argument('--language', default='italian')

    args = parser.parse_args()
    phonemizer = BACKENDS['espeak'](
        language2code[args.language],
        with_stress=True
    )
    blockify_dataset(args.language, args.path, args.outpath, phonemizer)
