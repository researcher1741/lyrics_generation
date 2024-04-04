from collections import defaultdict
from typing import List
import jsonlines
import itertools
import numpy as np
from tqdm import tqdm


def longest_substring(a, b):
    matrix = np.zeros((len(a), len(b)))
    result = 0
    for i in range(len(a)):
        for j in range(len(b)):
            if i == 0 or j == 0:
                matrix[i, j] = 0
            elif a[i - 1] == b[j - 1]:
                matrix[i, j] = matrix[i - 1, j - 1] + 1
                result = max(result, matrix[i, j])
            else:
                matrix[i, j] = 0
    return result


def do_rhyme(w1: str, w2: str, rhyming_dict: dict, use_similar_words=False):
    w1 = w1.lower()
    w2 = w2.lower()
    item1 = rhyming_dict.get(w1)
    if item1 is None:
        return False
    item2 = rhyming_dict.get(w2)
    if item2 is None:
        return False
    rhyme1 = item1['rhymes_set']
    rhyme2 = item2['rhymes_set']
    if use_similar_words:
        rhyme1.update(item1['similar_set'])
        rhyme2.update(item2['similar_set'])
    return w1 in rhyme2 or w2 in rhyme1


def add_rhyme_info(song, rhyming_dict, sentence_window=3, word_window=3, use_similar_to=False):
    lyrics: List = song['lyrics'].split('\n')
    rhyming_pairs = list()
    for i in range(sentence_window, len(lyrics)):
        current_sentence = lyrics[i].split()
        if len(current_sentence) == 0:
            continue
        current_interest_words = current_sentence[-word_window:]
        prevs = lyrics[i - sentence_window:i]
        for j, prev_sentence in enumerate(prevs):
            prev_sentence = prev_sentence.split()
            max_window = len(prev_sentence) // 2
            if max_window == 0:
                max_window = 1
            interest_words_window = min(word_window, max_window)
            prev_words = prev_sentence[-interest_words_window:]
            for combination in itertools.product(current_interest_words, prev_words):
                if do_rhyme(combination[0], combination[1], rhyming_dict, use_similar_to):
                    iw_idx = prev_words.index(combination[1])
                    ciw_idx = current_interest_words.index(combination[0])
                    rhyming_pairs.append({'sentence_1': i,
                                          'sentence_2': i - sentence_window + j,
                                          'sentence_1_idx': len(current_sentence) - word_window + ciw_idx,
                                          'sentence_2_idx': len(prev_sentence) - interest_words_window + iw_idx})
    song['rhymes'] = rhyming_pairs


def get_top_k(items, top_k):
    words = sorted([it for it in items if 'score' in it], key=lambda item: item['score'])
    if top_k > 0:
        words = {item['word'] for item in words[:top_k]}
    else:
        words = {}
    return words


def load_rhyming_dict(path, rhyming_top_k=-1, similar_to_top_k=-1):
    rhyming_dict = defaultdict(lambda x: None)
    with jsonlines.open(path) as items:
        for item in tqdm(items, desc='Loading rhyming dictionary'):
            word = item['word']
            del item['word']
            item['rhymes_set'] = get_top_k(item['rhymes'], rhyming_top_k)
            item['similar_set'] = get_top_k(item['sounds_like'], similar_to_top_k)
            rhyming_dict[word] = item
    return rhyming_dict


def main(path, rhyming_dict_path, outpath, sentence_window=3, word_window=3, rhyming_top_k=-1, similar_to_top_k=-1):
    rhyming_dict = load_rhyming_dict(rhyming_dict_path, rhyming_top_k=rhyming_top_k, similar_to_top_k=similar_to_top_k)
    use_similar = similar_to_top_k > 0
    with jsonlines.open(path) as lines, jsonlines.open(outpath, 'w') as writer:
        for song in tqdm(lines, desc='adding_rhyming_info'):
            add_rhyme_info(song, rhyming_dict, sentence_window, word_window, use_similar_to=use_similar)
            writer.write(song)


if __name__ == '__main__':
    path = 'data/wasabi/songs.with_phonetics.jsonl'
    rhyming_dict_path = 'data/wasabi/rhyming_dictionary.jsonl'
    outpath = 'data/wasabi/songs.with_phonetics.with_rhymes.sentence_window_3.word_window_3.rhyme_score_90.no_similar.jsonl'
    sentence_window = 3
    word_window = 3
    similar_to_top_k = -1
    rhyming_top_k = 25
    main(path, rhyming_dict_path, outpath, rhyming_top_k=rhyming_top_k, similar_to_top_k=similar_to_top_k)
