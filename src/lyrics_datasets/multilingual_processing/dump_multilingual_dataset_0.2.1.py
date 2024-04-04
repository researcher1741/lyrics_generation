import hashlib
from typing import List
# from src.lyrics_generation_utils.constants import *
from src.lyrics_generation_utils.constants import *
import jsonlines
import os
from tqdm import tqdm
import re


def clean_lyrics(lyrics, is_train):
    if lyrics is None:
        return None
    if len(lyrics[0]) == 1 and (lyrics[0][0].endswith(TAG_END) or len(lyrics[0]) == 1):
        lyrics = lyrics[1:]
    str_lyrics = ''.join([' '.join(x) for x in lyrics])
    if is_train:
        if '(' in str_lyrics or ')' in str_lyrics or 'x2' in str_lyrics.lower() or 'x 2' in str_lyrics.lower() or '[' in str_lyrics or ']' in str_lyrics:
            return None
    return lyrics


def format_lyrics(lyrics, schema, is_train):
    assert len(lyrics) == len(schema)
    new_lyrics = list()
    for letter, sentence in zip(schema, lyrics):
        if len(sentence) == 1:
            if is_train:
                return None
            else:
                new_lyrics.append(sentence[0])
                continue
        last_word = sentence[-2]
        i = -3
        if last_word.endswith(')'):
            while i > - len(sentence):
                last_word = sentence[i]
                i -= 1
                if last_word.startswith('('):
                    last_word = sentence[i]
                    break
        last_word = re.sub(r'[!@#$%^&\*\(\)\-_\+=|\\}{\[\]:;"\'?><.,]', '',
                           last_word).strip()  # strip off special symbols
        new_sentence = 'RHYME_' + letter + ' ' + last_word + SEP + ' ' + ' '.join(sentence).replace(' <sentence_end>',
                                                                                                    '<sentence_end>')
        new_lyrics.append(new_sentence)
    return new_lyrics


def build_input_output(line, is_train):
    artist = line['artist']
    title = line['title'].replace('_', ' ')
    lyrics: List[List[str]] = line['lyrics']
    genre = line['genre']
    if len(genre) == 0:
        genre = None
    topics = line['topics']
    if len(topics) == 0:
        topics = None
    emotions = line['emotions']
    if len(emotions) == 0:
        emotions = None
    rhyming_schema = line['rhyming_schema']
    prompt_lyrics = line['prompt_lyrics']
    lyrics = clean_lyrics(lyrics, is_train)
    prompt_lyrics = clean_lyrics(prompt_lyrics, is_train)
    language = line['lang']
    if lyrics is None:
        return None, None, None
    lyrics = format_lyrics(lyrics, rhyming_schema, is_train)
    if lyrics is None:
        return None, None, None
    lyrics = ''.join(lyrics)
    input_str = TITLE + title + ARTIST + artist
    if genre is not None:
        input_str += GENRE + genre
    if topics is not None:
        input_str += TOPICS + topics
    if emotions is not None:
        input_str += emotions
    if prompt_lyrics is not None:
        prompt_lyrics = ''.join([' '.join(s) for s in prompt_lyrics])
        input_str += LYRICS + prompt_lyrics
    input_str += LANG + language
    input_str += SCHEMA + ' '.join(['RHYME_' + x for x in rhyming_schema])
    output_str = lyrics
    return input_str, output_str, rhyming_schema


def dump_datasets(train_path, dev_path, test_path, outdir):
    keys = set()
    train_out = os.path.join(outdir, 'train.jsonl')
    dev_out = os.path.join(outdir, 'dev.jsonl')
    test_out = os.path.join(outdir, 'test.jsonl')
    print('dumping training')
    total_train = 0
    with jsonlines.open(train_path) as lines, jsonlines.open(train_out, 'w') as writer:
        for line in tqdm(lines):
            input_str, output_str, schema = build_input_output(line, True)
            if input_str is None:
                continue
            key = hashlib.md5((input_str + output_str).encode()).hexdigest()
            if key in keys:
                continue
            writer.write(
                {'key': key, 'input': input_str, 'output': output_str, 'schema': schema, 'language': line['lang']})
            keys.add(key)
            total_train += 1
    total_dev = 0
    print('dumping dev')
    with jsonlines.open(dev_path) as lines, jsonlines.open(dev_out, 'w') as writer:
        for line in tqdm(lines):
            input_str, output_str, schema = build_input_output(line, False)

            key = hashlib.md5((input_str + output_str).encode()).hexdigest()
            writer.write({'key': key, 'input': input_str, 'output': output_str, 'schema': schema})
            total_dev += 1

    print('dumping test')
    total_test = 0
    with jsonlines.open(test_path) as lines, jsonlines.open(test_out, 'w') as writer:
        for line in tqdm(lines):
            input_str, output_str, schema = build_input_output(line, False)
            key = hashlib.md5((input_str + output_str).encode()).hexdigest()
            writer.write({'key': key, 'input': input_str, 'output': output_str, 'schema': schema})
            total_test += 1

    print('done')
    print('total_train', total_train)
    print('total_dev', total_dev)
    print('total_test', total_test)


if __name__ == '__main__':
    dump_datasets('data/multilingual_section_dataset/train.jsonl',
                  'data/multilingual_section_dataset/dev.jsonl',
                  'data/multilingual_section_dataset/test.jsonl',
                  'data/multilingual_section_dataset_v0.2.1/')
