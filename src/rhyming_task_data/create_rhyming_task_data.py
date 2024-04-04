from argparse import ArgumentParser
import pronouncing
from collections import Counter, defaultdict

from sklearn.utils import shuffle
# from src.lyrics_generation_utils.constants import RHYME_TOKENS
from src.lyrics_generation_utils.constants import RHYME_TOKENS
import numpy as np
import jsonlines
import os
from tqdm import tqdm

def __random_schema(l):
    schema = np.random.choice(RHYME_TOKENS[:l], l, replace=True)
    return schema

def __sample_word(vocabulary:Counter):
    words = list(vocabulary.keys())
    probs = list(vocabulary.values())
    return np.random.choice(words, 1, p=probs)

def __get_rhyming_word(words, vocabulary:Counter):
    all_possible_rhymes = list()
    for w in words:
        rhymes = pronouncing.rhymes(w)
        all_possible_rhymes.append(rhymes - words)
    all_possible_rhymes = sorted(all_possible_rhymes, key=lambda x : len(x), reverse=True)
    intersection = set(all_possible_rhymes[0])
    for rhymes in all_possible_rhymes[1:]:
        intersection = intersection & rhymes
    
    rhyming_words = list(intersection)
    rhymes_probs = np.array([vocabulary[w] for w in intersection])
    norm_rhyme_probs = np.exp(rhymes_probs) / np.sum(np.exp(rhymes_probs))
    rhyming_word = np.random.choice(rhyming_words, 1, p=norm_rhyme_probs)
    return rhyming_word

def __fill_schema(schema, vocabulary:Counter):
    new_schema = defaultdict(set)
    filling = []
    for letter in schema:
        words = new_schema[letter]
        if len(words)==0:
            word = __sample_word(vocabulary)
            words.add(word)
            filling.append(word)
        else:
            word = __get_rhyming_word(words)
            filling.append(word)
            words.add(word)
    return filling


def create_data(num_items, vocabulary:Counter, max_len=10, **kwargs):
    items = list()
    min_len = 2
    for _ in tqdm(range(num_items), total=len(num_items)):
        schema_size = np.random.random_integers(min_len, max_len, 1)
        schema = __random_schema(schema_size)
        words = __fill_schema(schema, vocabulary)
        first_time = set()
        inputs = list()
        labels = list()
        for l, w in zip(schema, words):
            if l in first_time:
                labels.append(w)
                inputs.append(l + ' <mask>')
            else:
                inputs.append(l + ' ' + w)
                first_time.add(l)
        items.append((inputs, labels))
    return items

def split_data(data, num_training_instances, num_dev_instances, out_dir, **kwargs):
    shuffle(data)
    training_data = data[:num_training_instances]
    dev_data = data[num_training_instances:num_training_instances+num_dev_instances]
    test_data = data[num_training_instances+num_dev_instances:]
    for data, split_name in [(training_data, 'train'), (dev_data, 'dev'), (test_data, 'test')]:
        print(f'writing {split_name}')
        with jsonlines.open(os.path.join(out_dir, split_name + '.jsonl'), 'w') as writer:
            jsonl_data = [{'input':x, 'labels': y} for x, y in data]
            writer.write(jsonl_data)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_items', required=True)
    parser.add_argument('--vocabulary_path', required=True)
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--num_training_instances', required=True)
    parser.add_argument('--num_dev_instances', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    
    data = create_data(**vars(args))
    split_data(data, **args)