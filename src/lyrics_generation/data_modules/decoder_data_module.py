from copy import copy, deepcopy
from dataclasses import dataclass
from omegaconf import DictConfig
from transformers import DataCollatorForLanguageModeling
from src.lyrics_datasets.lyrics_blocks_dataset import LyricsBlockDataset
from torch.utils.data import DataLoader
from src.lyrics_generation.data_modules.lyrics_data_module import LyricsDataModule
from torch.nn.utils.rnn import pad_sequence
import torch

# from src.lyrics_generation_utils.constants import LYRICS
from src.lyrics_generation_utils.constants import LYRICS


@dataclass
class DecoderDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples, return_tensors=None):
        lyrics_start_token = self.tokenizer.encode(LYRICS, add_special_tokens=False)
        input_ids = [torch.LongTensor(e['input_ids'][:-1] + lyrics_start_token + e['labels'][1:]) for e in examples]
        input_ids = self.tokenizer.pad({'input_ids': input_ids})['input_ids']
        labels = input_ids.clone()
        labels.masked_fill_(labels == self.tokenizer.pad_token_id, -100).contiguous()

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


@dataclass
class GenerationDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples, return_tensors=None):
        lyrics_start_token = self.tokenizer.encode(LYRICS, add_special_tokens=False)
        input_ids = [torch.LongTensor(e['input_ids'][:-1] + lyrics_start_token) for e in examples]
        input_ids = self.tokenizer.pad({'input_ids': input_ids})['input_ids']
        labels = input_ids.clone()
        labels.masked_fill_(labels == self.tokenizer.pad_token_id, -100).contiguous()

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class DecoderDataModule(LyricsDataModule):
    def __init__(self, tokenizer, conf: DictConfig, data_collator=None, decoder_start_token_id=None):
        super().__init__(tokenizer, conf)
        if data_collator is None:
            data_collator = DecoderDataCollator(tokenizer, mlm=False)
        self.training_data_collector = data_collator
        self.validation_data_collector = data_collator
        self.decoder_start_token_id = decoder_start_token_id

    def get_dataset(self, conf, paths, split_name, tokenizer, limit, **kwargs):
        self.logger.info(f'{split_name.upper()} Data: {paths}')

        def song_filter(song):
            cond = song['lyrics_lengths'] + song['prompt_lengths'] < self.conf.data.max_len
            schema = song['prompt_schema']
            return cond and len(schema) > 0

        return LyricsBlockDataset(paths, conf.data.max_len, tokenizer,
                                  language=conf.data.language,
                                  dataset_name=conf.data.dataset_name,
                                  split_name=split_name,
                                  genre_mapping_path=conf.data.genre_mapping_path if hasattr(conf.data,
                                                                                             'genre_mapping_path') else None,
                                  year_mapping_path=conf.data.year_mapping_path if hasattr(conf.data,
                                                                                             'year_mapping_path') else None,
                                  limit=limit,
                                  rhyming_schema_dropout_probability=conf.data.rhyming_schema_dropout_probability if hasattr(
                                      conf.data, 'rhyming_schema_dropout_probability') else 0.0,
                                  decoder_start_token_id=self.decoder_start_token_id,
                                  song_filter=song_filter)

    def generation_dataloader(self):
        if not hasattr(self, 'test_data'):
            test_dataset = self.get_dataset(self.conf, self.conf.data.test_path, split_name='test',
                                            tokenizer=self.tokenizer, limit=10_000,
                                            prompt_component_probability=1.0)
        else:
            test_dataset = self.test_dataset
        if hasattr(self.validation_data_collector, 'tokenizer'):
            tokenizer = copy(self.tokenizer)
            tokenizer.padding_side = 'left'
        collator = GenerationDataCollator(tokenizer, mlm=False)

        return DataLoader(test_dataset,
                          batch_size=self.conf.data.batch_size,
                          num_workers=self.conf.data.num_workers,
                          collate_fn=collator)
