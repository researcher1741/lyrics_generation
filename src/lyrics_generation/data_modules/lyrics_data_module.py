from typing import Any, Union, List

from omegaconf import DictConfig
import torch
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.lyrics_datasets.lyrics_dataset import LyricsDataset

from src.lyrics_generation_utils.utils import get_info_logger, get_lyrics_tokenizer


class DataCollatorForLanguageModelingForTesting(DataCollatorForLanguageModeling):

    def __call__(self, features, return_tensors=None):
        new_features = []
        golds = []
        for f in features:
            input_ids = f['input_ids']
            input_length = f['prompt_len']
            if input_length == 0:
                input_length = len(input_ids)
            new_features.append({'input_ids': input_ids[:input_length], 'prompt_len': input_length})
            golds.append(input_ids[input_length:])
        ret = super().__call__(new_features, return_tensors)
        ret['gold_ids'] = golds
        return ret


class LyricsDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, conf: DictConfig):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.logger = get_info_logger(__name__)
        tok = get_lyrics_tokenizer(self.conf)
        tok.padding_side = 'left'
        self.validation_data_collector = DataCollatorForLanguageModeling(tok, mlm=False)
        self.training_data_collector = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def get_dataset(self, conf, paths, split_name, tokenizer, limit, prompt_component_probability, **kwargs):
        self.logger.info(f'{split_name.upper()} Data: {paths}')
        return LyricsDataset(paths, conf.data.max_len, tokenizer,
                             language=conf.data.language,
                             dataset_name=conf.data.dataset_name,
                             split_name=split_name,
                             genre_mapping_path=conf.data.genre_mapping_path,
                             year_mapping_path=conf.data.year_mapping_path,
                             prompt_component_probability=prompt_component_probability,
                             limit=limit,
                             with_prompt=conf.data.with_prompt,
                             **kwargs)

    def setup(self, *args, **kwargs):
        split_to_setup = ['train', 'test', 'dev']
        if 'split_to_setup' in kwargs:
            split_to_setup = kwargs['split_to_setup']
        self.logger.info('Preparing datasets')
        conf = self.conf
        tokenizer = self.tokenizer
        data_dict = vars(conf.data)['_content']
        data_dict['limit'] = -1
        self.train_dataset, self.dev_dataset, self.test_dataset = None, None, None
        if 'train' in split_to_setup:
            self.training_dataset = self.get_dataset(conf, conf.data.train_path, split_name='train',
                                                     tokenizer=tokenizer,
                                                     **data_dict
                                                     )
        data_dict['limit'] = 10_000
        data_dict['prompt_component_probability'] = 1.0
        if 'dev' in split_to_setup:
            self.dev_dataset = self.get_dataset(conf, conf.data.validation_path, split_name='dev', tokenizer=tokenizer,
                                                **data_dict)
        if 'test' in split_to_setup:
            self.test_dataset = self.get_dataset(conf, conf.data.test_path, split_name='test', tokenizer=tokenizer,
                                                 **data_dict)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.training_dataset, batch_size=self.conf.data.batch_size,
                          num_workers=self.conf.data.num_workers,
                          collate_fn=self.training_data_collector, shuffle=False)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dev_dataset, batch_size=self.conf.data.test_batch_size,
                          num_workers=self.conf.data.num_workers,
                          collate_fn=self.training_data_collector)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.conf.data.test_batch_size,
                          num_workers=self.conf.data.num_workers,
                          collate_fn=self.validation_data_collector)

    def generation_dataloader(self):
        test_dataset = self.test_dataset
        if test_dataset is None:
            test_dataset = self.get_dataset(self.conf, self.conf.data.test_path, split_name='test',
                                            tokenizer=self.tokenizer, limit=10_000,
                                            prompt_component_probability=1.0)

        tok = get_lyrics_tokenizer(self.conf)
        tok.padding_side = 'left'
        data_collectors = DataCollatorForLanguageModelingForTesting(tok, mlm=False)
        return DataLoader(self.test_dataset, batch_size=self.conf.data.batch_size,
                          num_workers=self.conf.data.num_workers,
                          collate_fn=data_collectors)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx=None) -> Any:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            if isinstance(v, dict):
                batch[k] = self.transfer_batch_to_device(v, device)
        return batch
