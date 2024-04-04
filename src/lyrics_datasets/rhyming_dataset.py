from random import shuffle
import pronouncing
import numpy as np
from collections import Counter
import pickle as pkl
import math


class EnRhymesDataset:
    def __init__(self, tokenizer, split_name, word_frequency_data_path=None, word_frequency_counter=None,
                 decoder_start_token_id=None) -> None:
        assert word_frequency_counter is not None or word_frequency_data_path is not None
        pronouncing.init_cmu()
        self.decoder_start_token_id = decoder_start_token_id
        self.word_freq = Counter()
        if word_frequency_counter is not None:
            self.word_freq = word_frequency_counter
        elif word_frequency_data_path is not None:
            with open(word_frequency_data_path, 'rb') as reader:
                self.word_freq = pkl.load(reader)
        words = [w for w in self.word_freq.keys() if
                 len([r for r in pronouncing.rhymes(w) if len(r) >= 3]) > 0 and len(w) >= 3]
        train_size = math.ceil(len(words) * 0.8)
        dev_size = math.ceil(len(words) * 0.1)
        if split_name == 'train':
            self.en_words = words[:train_size]
        elif split_name == 'dev':
            self.en_words = words[train_size:train_size + dev_size]
        else:
            self.en_words = words[train_size + dev_size:]

        shuffle(words)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        word = self.en_words[idx]
        possible_rhymes = [r for r in pronouncing.rhymes(word) if len(r) >= 3]
        freqs = np.array([self.word_freq[w] if w in self.word_freq else 1 for w in possible_rhymes])
        probs = freqs / np.sum(freqs)  # normalise to distribution
        chosen_words = np.random.choice(possible_rhymes, size=min(len(possible_rhymes), 3), replace=False, p=probs)
        input_str = 'rhyming word: ' + word
        output_str = 'words: ' + ', '.join(chosen_words)
        input_ids = self.tokenizer.encode(input_str)
        labels = self.tokenizer.encode(output_str, add_special_tokens=False)
        labels = [self.decoder_start_token_id] + labels
        if self.tokenizer.eos_token_id is not None:
            labels += [self.tokenizer.eos_token_id]
        relative_position_ids = list(reversed(list(range(len(labels) - 1, -1, -1))))
        rhyme_token_ids = [0] * len(labels)
        return {'input_ids': input_ids,
                'labels': labels,
                'relative_position_ids': relative_position_ids,
                'rhyme_token_ids': rhyme_token_ids,
                'task_name': 'rhyming word'
                }

    def __len__(self):
        return len(self.en_words)


class EnLastSyllablePronounceDataset:
    def __init__(self, tokenizer, split_name, decoder_start_token_id=None) -> None:
        pronouncing.init_cmu()
        words = [k for k, _ in pronouncing.pronunciations]
        train_size = math.ceil(len(words) * 0.8)
        dev_size = math.ceil(len(words) * 0.1)
        self.decoder_start_token_id = decoder_start_token_id
        if split_name == 'train':
            self.words = words[:train_size]
        elif split_name == 'dev':
            self.words = words[train_size:train_size + dev_size]
        else:
            self.words = words[train_size + dev_size:]
        self.tokenizer = tokenizer
        shuffle(self.words)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        phones = pronouncing.phones_for_word(word)[0]
        rhyming_phones = pronouncing.rhyming_part(phones)
        input_str = 'rhyming phone: ' + word
        output_str = 'phone: ' + rhyming_phones
        input_ids = self.tokenizer.encode(input_str)
        labels = self.tokenizer.encode(output_str, add_special_tokens=False)
        labels = [self.decoder_start_token_id] + labels
        if self.tokenizer.eos_token_id is not None:
            labels = labels + [self.tokenizer.eos_token_id]
        relative_position_ids = list(reversed(list(range(len(labels) - 1, -1, -1))))
        rhyme_token_ids = [self.tokenizer.pad_token_id] * len(labels)
        return {'input_ids': input_ids,
                'labels': labels,
                'relative_position_ids': relative_position_ids,
                'rhyme_token_ids': rhyme_token_ids,
                'task_name': 'rhyming phone'}
