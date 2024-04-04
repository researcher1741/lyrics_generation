from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from random import random
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer
from src.lyrics_datasets.lyrics_dataset import LyricsDataset
from src.lyrics_generation_utils.constants import *
from urllib.parse import unquote
import html
from torch.nn.utils.rnn import pad_sequence

from src.lyrics_generation_utils.constants_t5 import SCHEMA, RHYME_TOKENS
from src.lyrics_generation_utils.utils import get_lyrics_tokenizer
import torch
import numpy as np
import re
import pronouncing


@dataclass
class MultitaskDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples, return_tensors=None):
        input_ids, labels, decoder_rhyme_ids, decoder_relative_position_ids, task_names, languages = \
            zip(*[(torch.LongTensor(e['input_ids']),
                   torch.LongTensor(e['labels']),
                   torch.LongTensor(e['rhyme_token_ids']) if e['rhyme_token_ids'] is not None else None,
                   torch.LongTensor(e['relative_position_ids']) if e['relative_position_ids'] is not None else None,
                   e['task_name'],
                   e.get('language', None))
                  for e in examples])
        task_names = set(task_names)
        assert len(task_names) == 1 or print(
            'ERROR: It is not supported to have more than one task within the same batch. Tasks found:', task_names)
        task_name = list(task_names)[0]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        if decoder_rhyme_ids[0] is not None:
            decoder_rhyme_ids = pad_sequence(decoder_rhyme_ids, batch_first=True,
                                             padding_value=self.tokenizer.pad_token_id)
            decoder_rhyme_ids = decoder_rhyme_ids[:, :-1].contiguous()

        if decoder_relative_position_ids[0] is not None:
            decoder_relative_position_ids = pad_sequence(decoder_relative_position_ids, batch_first=True,
                                                         padding_value=0)
            decoder_relative_position_ids = decoder_relative_position_ids[:, :-1].contiguous()
        schemas = None
        if 'clear_schema' in examples[0]:
            schemas = [example['clear_schema'] for example in examples]
        decoder_input_ids = torch.masked_fill(labels[:, :-1], labels[:, :-1] == -100,
                                              self.tokenizer.pad_token_id).contiguous()
        labels = labels[:, 1:].contiguous()
        if all([x is None for x in languages]):
            languages = None
        return {'input_ids': input_ids,
                'labels': labels,
                'decoder_input_ids': decoder_input_ids,
                'decoder_rhyming_ids': decoder_rhyme_ids,
                'decoder_position_ids': decoder_relative_position_ids,
                'schema': schemas,
                'task_name': task_name,
                'languages': languages
                }


class LyricsBlockDataset(LyricsDataset):
    def __init__(self, path_or_paths,
                 max_len,
                 tokenizer: PreTrainedTokenizer,
                 dataset_name,
                 split_name,
                 genre_mapping_path,
                 year_mapping_path,
                 language=None,
                 num_processes=16,
                 limit=-1,
                 rhyming_schema_dropout_probability=0.5,
                 decoder_start_token_id=None,
                 song_filter=None,
                 version='0.2',
                 **kwargs
                 ) -> None:
        # tokenizer.add_prefix_space=True
        self.rhyming_schema_dropout_probability = rhyming_schema_dropout_probability
        self.decoder_start_token_id = decoder_start_token_id
        self.lyrics_start_ids = tokenizer.encode(LYRICS, add_special_tokens=False)
        self.space_special_token = tokenizer.tokenize(' .')[0].replace('.', '')
        self.special_tokens_set = set(LYRICS_SPECIAL_TOKENS)
        self.add_language = kwargs.get('add_language', False)

        super().__init__(path_or_paths, max_len, tokenizer, dataset_name, split_name,
                         genre_mapping_path, year_mapping_path,
                         language, num_processes, 1.0, limit, True, version=version)
        if song_filter is None:
            song_filter = self._default_song_filter
        self.examples = self.examples.filter(song_filter, num_proc=16)
        self.logger.info(f'{split_name}: {len(self.examples)}')

    def _default_song_filter(self, song):
        if self.split_name == 'train':
            cond = song['lyrics_lengths'] < self.max_len and song['prompt_lengths'] < self.max_len
            schema = song['rhyming_schema']
            return cond and len(schema) > 0
        return True

    def _get_filtering_function(self):
        if self.split_name == 'train':
            # filters all items where there is no rhyme
            return lambda example: len(example['rhyming_schema']) > len(set(example['rhyming_schema']))
        return None

    def get_song_topics(self, topics):
        return [x.lower() for x in topics.split(',') if
                len(x) > 0 and x.lower() not in {'does', 'doesn', 'didn', 'would', 'nigga', 'niggas', 'wouldn', 'could',
                                                 'couldn'}]

    def get_topics(self, songs):
        if 'topics' in songs:
            ## TODO should improve filtering with POS tags and offensive word vocabulary
            topics = []
            for t in songs['topics']:
                topics.append(self.get_song_topics(t))
        else:
            topics = []

        return topics

    def get_emotion(self, song_emotion):
        if song_emotion:
            return [SEMI_AUTOMATIC_REVERSE_CLUSTERING[x.strip().lower()] for x in song_emotion if
                    len(x.strip()) > 0 and x.strip().lower() in SEMI_AUTOMATIC_REVERSE_CLUSTERING]
        return None

    def get_emotions(self, songs):
        if 'emotions' in songs:
            emotions = [self.get_emotion(e) for e in songs['emotions']]  # list of lists deal with it properly
        else:
            emotions = None
        return emotions

    def reverse_flat(self, lst_of_lst, sentences_per_block):
        start = 0
        lyrics = []
        for num_sentences in sentences_per_block:
            block = lst_of_lst[start:start + num_sentences]
            lyrics.append(block)
            start += num_sentences
        return lyrics

    def tokenize_lyrics(self, songs, lyrics_key, is_prompt):
        # if is_prompt and isinstance(songs[lyrics_key][0], str):
        # special case for chinese dataset that doesn't have prompt split in blocks and tokens
        #     return self.tokenizer(songs[lyrics_key], add_special_tokens=False).encodings
        lyrics = songs[lyrics_key]
        lyrics = [b if b is not None else [] for b in lyrics]
        num_blocks = [len(x) for x in lyrics]
        flat_lyrics = [s for block in lyrics for s in block]
        flat_lyrics = [' '.join(s).replace(' ' + SENTENCE_END, SENTENCE_END) for s in flat_lyrics]
        if sum(num_blocks) == 0:
            flat_encoded_lyrics = [None] * len(num_blocks)
        else:
            flat_encoded_lyrics = self.tokenizer(flat_lyrics, add_special_tokens=False)

        encoded_lyrics = self.reverse_flat(flat_encoded_lyrics, num_blocks)
        return encoded_lyrics

    def _normalise_title(self, title):
        if '(' in title:
            title = title[:title.index('(')]
        if '[' in title:
            title = title[:title.index('[')]
        return title

    def _batch_tokenise_lyrics(self, songs):
        raw_artists = songs['artist']
        raw_titles = songs['title']
        raw_genres = songs['genre']
        raw_n_syllables = songs['num_syllables']
        raw_languages = songs['lang']
        raw_topics = songs['topics']
        raw_emotions = songs['emotions']
        artists, titles, genres, num_syllables, languages = [], [], [], [], []
        for i in range(len(songs['title'])):
            artists.append(ARTIST + html.unescape(unquote(','.join(raw_artists[i]))))
            titles.append(TITLE + self._normalise_title(html.unescape(unquote(raw_titles[i].replace('_', ' ')))))
            genres.append(GENRE + html.unescape(unquote(raw_genres[i])))
            num_syllables.append(NUM_SYLLABLES + ','.join([str(x) for x in raw_n_syllables[i]]))
            languages.append(LANG + raw_languages[i])
        aux = artists + titles + genres + languages + num_syllables
        encoded_aux = self.tokenizer(aux, add_special_tokens=False)['input_ids']
        category_len = len(artists)
        encoded_artists = encoded_aux[:category_len]
        encoded_titles = encoded_aux[category_len:2 * category_len]
        encoded_genres = [x if len(x) > 1 else [] for x in encoded_aux[2 * category_len:3 * category_len]]
        encoded_languages = encoded_aux[3 * category_len:4 * category_len]
        encoded_num_syllables = encoded_aux[4 * category_len: 5 * category_len]
        encoded_topics = [
            self.tokenizer(t.split(','), add_special_tokens=False)['input_ids'] if len(t) > 0 else [[-100]] for t in
            raw_topics] if raw_topics else None
        encoded_emotions = [
            self.tokenizer(e.split(','), add_special_tokens=False)['input_ids'] if len(e) > 0 else [[-100]] for e in
            raw_emotions] if raw_emotions else None
        prompt_encoded_lyrics = self.tokenize_lyrics(songs, 'prompt_lyrics', is_prompt=True)
        encoded_lyrics = self.tokenize_lyrics(songs, 'lyrics', is_prompt=False)
        encoded_lyrics_ids = []
        encoded_lyrics_word_ids = []
        encoded_lyrics_relative_positions = []
        encoded_schemas = []
        prompt_encoded_lyrics_ids = []
        prompt_encoded_word_ids = []
        rhyming_schema = songs['rhyming_schema']
        for l, rs, pl in zip(encoded_lyrics, rhyming_schema, prompt_encoded_lyrics):
            lyrics_ids = self.build_input(l)
            prompt_lyrics_ids = self.build_input(pl)
            encoded_lyrics_ids.append(lyrics_ids['input_ids'])
            encoded_lyrics_word_ids.append(lyrics_ids['word_ids'])
            encoded_lyrics_relative_positions.append(lyrics_ids['relative_positions'])
            prompt_encoded_lyrics_ids.append(prompt_lyrics_ids['input_ids'])
            prompt_encoded_word_ids.append(prompt_lyrics_ids['word_ids'])
            max_rhyming_letter = max([ord(x) for x in rs])
            if max_rhyming_letter > ord('Z'):
                encoded_schemas.append([])
            else:
                rhyme_tokens = SCHEMA + ' '.join([RHYME_TOKENS[ord(x) - ord('A')] for x in rs])
                rhyme_tokens = self.tokenizer.encode(rhyme_tokens, add_special_tokens=False)
                encoded_schemas.append(rhyme_tokens)
        lyrics_lengths = [len(l) + len(t) + len(a) + len(g) + sum(len(x) for x in em) + sum(len(x) for x in to) for
                          l, t, a, g, em, to in
                          zip(encoded_lyrics_ids, encoded_titles, encoded_artists, encoded_genres, encoded_emotions,
                              encoded_topics)]
        prompt_lengths = [len(x) for x in prompt_encoded_lyrics_ids]
        ret = {
            'lyrics_lengths': lyrics_lengths,
            'prompt_lengths': prompt_lengths,
            'encoded_lyrics': encoded_lyrics_ids,
            'encoded_title': encoded_titles,
            'encoded_artist': encoded_artists,
            'encoded_genre': encoded_genres,
            'encoded_emotion_tags': encoded_emotions,
            'encoded_topics': encoded_topics,
            'prompt_schema': encoded_schemas,
            'rhyming_schema': rhyming_schema,
            'encoded_num_syllables': encoded_num_syllables,
            'encoded_lyrics_word_ids': encoded_lyrics_word_ids,
            'encoded_lyrics_relative_positions': encoded_lyrics_relative_positions,
            'prompt_encoded_lyrics': prompt_encoded_lyrics_ids,
            'prompt_encoded_word_ids': prompt_encoded_word_ids,
            'encoded_language': encoded_languages
        }
        ret = {k: v for k, v in ret.items() if v is not None}
        return ret

    def build_input(self, encodings):
        word_mapping = [e.word_ids for e in encodings]
        relative_positions = [list(reversed(range(len(e.ids)))) for e in encodings]
        ids = [i for e in encodings for i in e.ids]
        relative_positions = [p for rp in relative_positions for p in rp]
        flat_word_mapping = []
        count = 0
        for sentence_wi in word_mapping:
            for wi in sentence_wi:
                flat_word_mapping.append(wi + count)
            count += len(set(sentence_wi))
        return {'input_ids': ids, 'word_ids': flat_word_mapping, 'relative_positions': relative_positions}

    def sample_from_collection(self, collection, tag_to_append, min_sample_size):
        if collection is not None and len(collection) > 0:
            if len(collection) == 1:
                sample_size = 1
            else:
                sample_size = np.random.randint(1, min(min_sample_size + 1, len(collection)), 1)[0] if len(
                    collection) > 0 else None
            sample_idxs = np.random.choice(len(collection), sample_size, replace=False)
            sample = [collection[i] for i in sample_idxs]
            aux = []
            for j, s in enumerate(sample):
                aux.extend(s)
                if j < len(sample) - 1:
                    aux.extend(self.encoded_comma)
            return tag_to_append + aux
        return None

    def build_schema_pairs(self, schema):
        same_rhyme = defaultdict(list)
        for i, val in enumerate(schema):
            same_rhyme[val].append(i)
        pairs = list()
        for l in same_rhyme.values():
            if len(l) < 2:
                continue
            for i in range(len(l)):
                for j in range(i + 1, len(l)):
                    pairs.append((l[i], l[j]))
        pairs_set = set(pairs)
        negative_pairs = set()
        for i in range(len(schema)):
            for j in range(i + 1, len(schema)):
                if (i, j) not in pairs_set and (j, i) not in pairs_set:
                    negative_pairs.add((i, j))
        return list(pairs), list(negative_pairs)

    def get_last_phoneme(word: str):
        phonemes = pronouncing.phones_for_word(word).split(' ')
        idx = len(phonemes) - 1
        for i in range(len(phonemes) - 1, -1, -1):
            if phonemes[i][-1] in '012':  # if it is a vowel
                return ' '.join(phonemes[i:])
        return phonemes

    def __getitem__(self, index):
        example = self.examples[index]
        title = example['encoded_title']
        artist = example['encoded_artist']
        input_lyrics = example['prompt_encoded_lyrics']
        input_schema = example['prompt_schema']
        genre = example['encoded_genre']
        language = example['encoded_language']
        emotions = [x for x in example['encoded_emotion_tags'] if len(x) > 0 and x != [-100]]
        topics = [x for x in example['encoded_topics'] if len(x) > 0 and x != [-100]]
        input_ids = []
        # if random() >=0.5 or self.split_name != 'train':
        input_ids += title
        if random() >= 0.5 or self.split_name != 'train':
            input_ids += artist
        if self.tokenizer.bos_token_id is not None:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
        if len(genre) > 0:
            input_ids += genre
        if len(emotions) > 0:
            emotions = self.sample_from_collection(emotions, self.encoded_emotion_tag, 2)
            input_ids.extend(emotions)
        if len(topics) > 0:
            topics = self.sample_from_collection(topics, self.encoded_topics_tag, 3)
            input_ids.extend(topics)

        clear_schema = example['rhyming_schema']
        encoded_lyrics = self._build_label(example['encoded_lyrics'], copy(clear_schema))
        sentence_end_id = self.tokenizer.encode(SENTENCE_END, add_special_tokens=False)[0]
        lyrics_str = self.tokenizer.convert_ids_to_tokens(encoded_lyrics)
        sentence_end_idxs = np.where(encoded_lyrics == sentence_end_id)[0]
        encoded_lyrics = list(encoded_lyrics)

        if len(input_lyrics) > 0:
            input_ids += self.lyrics_start_ids + input_lyrics

        output_lyrics = encoded_lyrics
        sentence_end_idxs -= 1  # subtract 1 to take indices of last tokens
        prev_rhyming_token_idxs = []
        rhyme_ids = []
        aux = []
        for idx in sentence_end_idxs:  # find index of last non-symbol token (which is the rhyming token)
            strpiece = lyrics_str[idx].replace(self.space_special_token, '')
            while re.match(r'[\[\]}{\'":;!/?>.,<~`@#$%^&*(\|)_\-+=0987654321]+', strpiece) is not None:
                idx -= 1
                strpiece = lyrics_str[idx].replace(self.space_special_token, '')
            aux.append(idx)
            while not lyrics_str[idx].startswith(self.space_special_token) and lyrics_str[
                idx] not in self.special_tokens_set:
                idx -= 1
            prev_rhyming_token_idxs.append(idx - 1)
        rhyme_ids = [x.replace('RHYME_', '') for x in lyrics_str if 'RHYME_' in x]
        sentence_end_idxs = aux
        rhyme_token_ids = np.zeros_like(encoded_lyrics) + self.tokenizer.pad_token_id
        rhyme_token_ids[np.array(prev_rhyming_token_idxs)] = [
            self.tokenizer.encode(RHYME_TOKENS[ord(x) - ord('A')], add_special_tokens=False)[0] for x in rhyme_ids]
        if self.decoder_start_token_id is not None:
            rhyme_token_ids = np.concatenate([np.zeros(1), rhyme_token_ids], 0)
        if self.tokenizer.eos_token_id is not None:
            rhyme_token_ids = np.concatenate([rhyme_token_ids, np.zeros(1)], 0)

        relative_position_ids = []
        input_ids += input_schema
        if self.add_language:
            input_ids += language
        if self.tokenizer.eos_token_id is not None:
            input_ids += [self.tokenizer.eos_token_id]

        labels = []
        if self.decoder_start_token_id is not None:
            labels = [self.decoder_start_token_id]
        labels += output_lyrics
        if self.tokenizer.eos_token_id is not None:
            labels += [self.tokenizer.eos_token_id]
        # we don't care about the pos_id for eos_token since it won't be given as input to the decoder.
        relative_position_ids = [0, 0]
        k = 1
        for x in reversed(labels[:-2]):
            if x == sentence_end_id:
                k = 0
            relative_position_ids.insert(0, k)
            k += 1
        assert len(labels) == len(rhyme_token_ids) == len(relative_position_ids)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'rhyme_token_ids': rhyme_token_ids,
            'relative_position_ids': relative_position_ids,
            'clear_schema': clear_schema,
            'task_name': 'lyrics generation',
            'language': language
        }

    def _build_label(self, encoded_lyrics, clear_schema):
        encoded_lyrics = np.array(encoded_lyrics)
        tag_end_id = self.tokenizer(TAG_END, add_special_tokens=False)['input_ids']
        sentence_end_id = self.tokenizer(SENTENCE_END, add_special_tokens=False)['input_ids']
        if any(encoded_lyrics == tag_end_id):
            idx = np.argwhere(encoded_lyrics == tag_end_id)[0][0]
            first_letter = clear_schema.pop(0)  # pop from the head
            rhyming_idx = ord(first_letter) - ord('A')
            rhyme_token_id = np.array(self.tokenizer(RHYME_TOKENS[rhyming_idx], add_special_tokens=False)['input_ids'])
            encoded_lyrics = np.concatenate([np.array(rhyme_token_id), encoded_lyrics[idx + 1:]], 0)
        for idx in reversed(np.argwhere(encoded_lyrics == sentence_end_id)[:-1]):
            idx = idx[0]
            letter = clear_schema.pop()  # pop from the tail because we are iterating indices in reverse orther
            rhyming_idx = ord(letter) - ord('A')
            rhyme_token_id = np.array(
                self.tokenizer(RHYME_TOKENS[rhyming_idx] + ' ', add_special_tokens=False)['input_ids'])
            encoded_lyrics = np.concatenate([encoded_lyrics[:idx + 1], rhyme_token_id, encoded_lyrics[idx + 1:]], 0)
        if len(clear_schema) > 0:
            letter = clear_schema.pop()  # pop from the tail because we are iterating indices in reverse orther
            rhyming_idx = ord(letter) - ord('A')
            rhyme_token_id = np.array(self.tokenizer(RHYME_TOKENS[rhyming_idx], add_special_tokens=False)['input_ids'])
            encoded_lyrics = np.concatenate([rhyme_token_id, encoded_lyrics], 0)
        return encoded_lyrics


if __name__ == '__main__':
    path = './LG/DATA/genius_section_0.2/test.10000.jsonl'
    tokenizer = get_lyrics_tokenizer('google/t5-v1_1-large')
    genres_mapping_path = 'data/genres_mapping_100.txt'
    dataset = LyricsBlockDataset(path, 300, tokenizer, 'genius', 'test', genres_mapping_path)

    dataset[0]
    print(len(dataset))
