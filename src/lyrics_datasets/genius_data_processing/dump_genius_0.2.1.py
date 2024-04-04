import hashlib
from typing import List
from lyrics_generation_utils.constants import * 
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
        last_word = re.sub(r'[!@#$%^&\*\(\)\-_\+=|\\}{\[\]:;"\'?><.,]', '', last_word).strip() # strip off special symbols 
        new_sentence = 'RHYME_' + letter + ' ' + last_word + SEP + ' ' + ' '.join(sentence).replace(' <sentence_end>', '<sentence_end>')
        new_lyrics.append(new_sentence)
    return new_lyrics

    

def build_input_output(line, is_train):
    artist = line['artist']
    title = line['title'].replace('_', ' ')
    lyrics:List[List[str]] = line['lyrics']
    genre = line['genre'].strip()
    topics = line['topics']
    emotions = line['emotions']
    rhyming_schema = line['rhyming_schema']
    prompt_lyrics = line['prompt_lyrics']
    lyrics = clean_lyrics(lyrics, is_train)
    prompt_lyrics = clean_lyrics(prompt_lyrics, is_train)
    if lyrics is None:
        return None, None, None
    lyrics = format_lyrics(lyrics, rhyming_schema, is_train)
    if lyrics is None:
        return None, None, None
    lyrics = ''.join(lyrics)
    input_str = TITLE + title + ARTIST + artist + SCHEMA + ' '.join(['RHYME_' + x for x in rhyming_schema])
    if genre is not None and len(genre.strip()) > 0:
        input_str += GENRE + genre
    if topics is not None and len(topics.strip()) > 0: 
        input_str += TOPICS + topics
    if emotions is not None and len(emotions.strip()) > 0:
        input_str += emotions
    if prompt_lyrics is not None:
        prompt_lyrics = ''.join([' '.join(s) for s in prompt_lyrics])
        input_str += LYRICS + prompt_lyrics
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
            writer.write({'key': key, 'input':input_str, 'output': output_str, 'schema': schema})
            keys.add(key)
            total_train += 1
    total_dev = 0
    print('dumping dev')
    keys = set()
    with jsonlines.open(dev_path) as lines, jsonlines.open(dev_out, 'w') as writer:
        for line in tqdm(lines):
            input_str, output_str, schema = build_input_output(line, False)
            key = hashlib.md5((input_str + output_str).encode()).hexdigest()
            if key in keys:
                continue
            keys.add(key)
            writer.write({'key': key, 'input':input_str, 'output': output_str, 'schema': schema, 'language':'english'})
            total_dev += 1

    print('dumping test')
    total_test = 0
    keys = set()
    with jsonlines.open(test_path) as lines, jsonlines.open(test_out, 'w') as writer:
        for line in tqdm(lines):
            input_str, output_str, schema = build_input_output(line, False)
            key = hashlib.md5((input_str + output_str).encode()).hexdigest()
            if key in keys:
                continue
            keys.add(key)
            writer.write({'key': key, 'input':input_str, 'output': output_str, 'schema': schema})
            total_test += 1
    
            
    print('done')
    print('total_train', total_train)
    print('total_dev', total_dev)
    print('total_test', total_test)
            

if __name__ == '__main__':
    dump_datasets('./LG/DATA/genius_section_0.2/train.jsonl',
                './LG/DATA/genius_section_0.2/dev.3500.jsonl',
                './LG/DATA/genius_section_0.2/test.3500.jsonl',
                'data/genius/phonemised_dataset/section_dataset_rhyme_and_verse_length_filtered_v0.2.1.1/')