from argparse import ArgumentParser
from collections import deque
from random import shuffle
import jsonlines
import os
import numpy as np
from tqdm import tqdm


def __load_data(path, language):
    data = []
    with jsonlines.open(path) as lines:
        for song in lines:
            song['lang'] = language
            data.append(song)
    return data


def dump_data(data, writer, keys, split_name):
    for song in tqdm(data, desc=split_name):
        if 'schema' in song:
            song['rhyming_schema'] = song['schema']
            del song['schema']
        else:
            assert 'rhyming_schema' in song
        if 'emotion_tags' in song:
            song['emotions'] = song['emotion_tags']
            del song['emotion_tags']
        if 'emotions' in song:
            if song['emotions'] is None or song['emotions'] == []:
                song['emotions'] = ''
            elif isinstance(song['emotions'], list):
                song['emotions'] = ','.join(song['emotions'])
        else:
            song['emotions'] = ''
        if 'topics' in song:
            if song['topics'] is None or song['topics'] == []:
                song['topics'] = ''
            elif isinstance(song['topics'], list):
                song['topics'] = ','.join(song['topics'])
        else:
            song['topics'] = ''
        song['num_syllables'] = []
        song['prompt_rhyming_schema'] = None
        for k in keys:
            if k not in song:
                song[k] = []
        writer.write(song)


def create_multilingual_dataset(english_train_path, english_dev_path, english_test_path, multilingual_folder,
                                out_dir_path):
    print('loading en training')
    en_train = __load_data(english_train_path, 'english')
    print('loading en dev')
    en_dev = __load_data(english_dev_path, 'english')
    print('loading en test ')
    en_test = __load_data(english_test_path, 'english')
    data = list()
    with jsonlines.open(os.path.join(out_dir_path, "train.jsonl"), 'w') as train_writer, \
            jsonlines.open(os.path.join(out_dir_path, 'dev.jsonl'), 'w') as dev_writer, \
            jsonlines.open(os.path.join(out_dir_path, 'test.jsonl'), 'w') as test_writer:
        keys = None
        for song in en_train + en_test + en_dev:
            if keys is None:
                keys = song.keys()
            data.append(song)
        for file in os.listdir(multilingual_folder):
            language = file.split('_')[0]
            print(language)
            if language != 'english':
                file = os.path.join(multilingual_folder, file)
                lang_data = __load_data(file, language)
                data.extend(lang_data)

        shuffle(data)

        dev_size = 10_000
        test_keys = set()
        test = list()
        while True:
            elem = data.pop()
            key = elem['artist'].lower() + '_' + elem['title'].lower()
            test_keys.add(key)
            test.append(elem)
            if len(test) == dev_size:
                break
        dev_keys = set()
        dev = list()
        while True:
            elem = data.pop()
            key = elem['artist'].lower() + '_' + elem['title'].lower()
            if key in test:
                continue
            dev_keys.add(key)
            dev.append(elem)
            if len(dev) == dev_size:
                break
        train = list()
        for elem in data:
            key = elem['artist'].lower() + '_' + elem['title'].lower()
            if key in test_keys or key in dev_keys:
                continue
            train.append(elem)
        dump_data(train, train_writer, keys, 'train')
        dump_data(dev, dev_writer, keys, 'dev')
        dump_data(test, test_writer, keys, 'test')

    # print('shuffling training')
    # os.system(f'shuf {train_temp_path} > {os.path.join(out_dir_path, "train.jsonl")}')
    print('Done')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--en_train_path',
                        default='./LG/DATA/genius_section_0.2/train.jsonl')
    parser.add_argument('--en_dev_path',
                        default='./LG/DATA/genius_section_0.2/dev.3500.jsonl')
    parser.add_argument('--en_test_path',
                        default='./LG/DATA/genius_section_0.2/test.3500.jsonl')
    parser.add_argument('--multilingual_dir_path',
                        default='/lyrics_generation/data/wasabi/songs_by_language/section_dataset')
    parser.add_argument('--out_dir_path',
                        default='/lyrics_generation/data/multilingual_section_dataset/')
    args = parser.parse_args()
    create_multilingual_dataset(args.en_train_path, args.en_dev_path, args.en_test_path, args.multilingual_dir_path,
                                args.out_dir_path)
