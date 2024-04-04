from typing import Any, Dict
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorMixin
import torch
import re
from random import random
import numpy as np
from src.lyrics_datasets.lyrics_blocks_dataset_last_word_first import LyricsBlockDatasetLastWordFirst
from src.lyrics_datasets.lyrics_blocks_dataset import LyricsBlockDataset
from src.lyrics_generation_utils.constants import RHYME_TOKENS, SENTENCE_END
from copy import copy

from src.lyrics_generation_utils.utils import get_lyrics_tokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification


class PretrainingDataCollator(DataCollatorMixin):
    def __init__(self, pad_token, pad_side) -> None:
        super().__init__()
        self.pad_token = pad_token
        self.pad_side = pad_side

    def __call__(self, all_input_ids) -> Dict[Any, torch.Tensor]:
        max_len = max([len(x) for x in all_input_ids])
        padded_input_ids = list()
        for input_ids in all_input_ids:
            pad_len = max_len - len(input_ids)
            padding_list = [self.pad_token] * pad_len
            if self.pad_side == 'right':
                input_ids = input_ids + padding_list
            else:
                input_ids = padding_list + input_ids
            padded_input_ids.append(input_ids)
        padded_input_ids = torch.LongTensor(padded_input_ids)
        return {'input_ids': padded_input_ids}


class PretrainingDatasetEncoderDecoder(LyricsBlockDataset):
    def __init__(self, path_or_paths, max_len, tokenizer: PreTrainedTokenizer, dataset_name, split_name,
                 genre_mapping_path,
                 language=None, num_processes=1, limit=-1, rhyming_schema_dropout_probability=0.5,
                 decoder_start_token_id=None, song_filter=None, version='0.2.1', **kwargs) -> None:
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.bos_token_id
        super().__init__(path_or_paths, max_len, tokenizer, dataset_name, split_name, genre_mapping_path,
                         language, num_processes, limit, rhyming_schema_dropout_probability, decoder_start_token_id,
                         song_filter, version, **kwargs)
        self.prompt_start = tokenizer.encode('<prompt>', add_special_tokens=False)[0]
        self.prompt_end = tokenizer.encode('</prompt>', add_special_tokens=False)[0]

    def __getitem__(self, index):
        example = self.examples[index]
        title = example['encoded_title']
        artist = example['encoded_artist']
        input_lyrics = example['prompt_encoded_lyrics']
        input_schema = example['prompt_schema']
        genre = example['encoded_genre']
        emotions = [x for x in example['encoded_emotion_tags'] if len(x) > 0]
        topics = [x for x in example['encoded_topics'] if len(x) > 0]
        language = self.tokenizer.decode(example['encoded_language']).split('>')[-1].strip()

        input_ids = [self.prompt_start]
        if random() >= 0.5 or self.split_name != 'train':
            input_ids += title
        if random() >= 0.5 or self.split_name != 'train':
            input_ids += artist
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
        encoded_lyrics = list(encoded_lyrics)

        if len(input_lyrics) > 0:
            input_ids += input_lyrics

        input_ids += input_schema + [self.prompt_end]

        labels = []
        labels += [self.tokenizer.bos_token_id] + encoded_lyrics + [self.tokenizer.eos_token_id]
        return {
            'input_ids': input_ids,
            'labels': labels,
            'rhyme_token_ids': None,
            'relative_position_ids': None,
            'clear_schema': clear_schema,
            'task_name': 'lyrics generation',
            'language': language
        }


class PretrainingDatasetEncoderDecoderLastWordFirst(LyricsBlockDatasetLastWordFirst):
    def __init__(self, path_or_paths, max_len, tokenizer: PreTrainedTokenizer, dataset_name, split_name,
                 genre_mapping_path,
                 language=None, num_processes=1, limit=-1, rhyming_schema_dropout_probability=0.5,
                 decoder_start_token_id=None, song_filter=None, version='0.2.2', **kwargs) -> None:
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.bos_token_id
        super().__init__(path_or_paths, max_len, tokenizer, dataset_name, split_name, genre_mapping_path,
                         language, num_processes, limit, rhyming_schema_dropout_probability, decoder_start_token_id,
                         song_filter, version, **kwargs)
        self.prompt_start = tokenizer.encode('<prompt>', add_special_tokens=False)[0]
        self.prompt_end = tokenizer.encode('</prompt>', add_special_tokens=False)[0]

    def __getitem__(self, index):
        example = self.examples[index]
        title = example['encoded_title']
        artist = example['encoded_artist']
        input_lyrics = example['prompt_encoded_lyrics']
        input_schema = example['prompt_schema']
        genre = example['encoded_genre']
        emotions = [x for x in example['encoded_emotion_tags'] if len(x) > 0 and x != [-100]]
        topics = [x for x in example['encoded_topics'] if len(x) > 0 and x != [-100]]
        language = self.tokenizer.decode(example['encoded_language']).split('>')[-1].strip()
        input_ids = [self.prompt_start]
        if random() >= 0.5 or self.split_name != 'train':
            input_ids += title
        if random() >= 0.5 or self.split_name != 'train':
            input_ids += artist
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
        encoded_lyrics = list(encoded_lyrics)

        if len(input_lyrics) > 0:
            input_ids += input_lyrics

        input_ids += input_schema + [self.prompt_end]

        labels = []
        if self.tokenizer.bos_token_id is not None:
            labels.append(self.tokenizer.bos_token_id)
        else:
            labels.append(self.tokenizer.pad_token_id)
        labels.extend(encoded_lyrics)
        if self.tokenizer.eos_token_id is not None:
            labels.append(self.tokenizer.eos_token_id)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'rhyme_token_ids': None,
            'relative_position_ids': None,
            'clear_schema': clear_schema,
            'task_name': 'lyrics generation',
            'language': language
        }


if __name__ == '__main__':
    path = './LG/DATA/genius_section_0.2/test.jsonl'
    tokenizer = get_lyrics_tokenizer('tokenizers/genius_section_tokenizer/')
    genres_mapping_path = 'data/genres_mapping_100.txt'
    dataset = PretrainingDatasetEncoderDecoder(path, 300, tokenizer, 'genius', 'test', genres_mapping_path)
    dataset[0]
    print(len(dataset))
