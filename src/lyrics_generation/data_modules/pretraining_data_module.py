from typing import Any, Union, List

from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.lyrics_datasets.pretraining_dataset import PretrainingDataCollator, PretrainingDataset
# from src.lyrics_generation_utils.constants import GENERATE, LYRICS
from src.lyrics_generation_utils.constants import GENERATE, LYRICS

from src.lyrics_generation_utils.utils import get_info_logger


class PretrainingDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, conf: DictConfig):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.logger = get_info_logger(__name__)
        self.data_version = conf.data.version
        self.lyrics_token_id = tokenizer.encode(LYRICS, add_special_tokens=False)[0]
        self.generation_data_collector = PretrainingDataCollator(self.tokenizer.pad_token_id, 'left')
        self.validation_data_collector = PretrainingDataCollator(self.tokenizer.pad_token_id, 'right')
        self.training_data_collector = PretrainingDataCollator(self.tokenizer.pad_token_id, 'right')

    def get_dataset(self, conf, paths, split_name, limit, for_generation=False, **kwargs):
        self.logger.info(f'{split_name.upper()} Data: {paths}')
        return PretrainingDataset(paths, conf.data.max_len,
                                  dataset_name=conf.data.dataset_name,
                                  split_name=split_name,
                                  limit=limit,
                                  version=conf.data.version,
                                  lyrics_token_id=self.lyrics_token_id,
                                  for_generation=for_generation
                                  )

    def setup(self, *args, **kwargs):
        split_to_setup = ['train', 'test', 'dev']
        if 'split_to_setup' in kwargs:
            split_to_setup = kwargs['split_to_setup']
        self.logger.info('Preparing datasets')
        conf = self.conf
        data_dict = vars(conf.data)['_content']
        if 'limit' not in data_dict:
            data_dict['limit'] = -1
        self.training_dataset, self.dev_dataset, self.test_dataset = None, None, None
        if 'train' in split_to_setup:
            self.training_dataset = self.get_dataset(conf, conf.data.train_path, split_name='train', **data_dict)
        if 'dev' in split_to_setup:
            data_dict['limit'] = 10_000
            self.dev_dataset = self.get_dataset(conf, conf.data.validation_path, split_name='dev', **data_dict)
        if 'test' in split_to_setup:
            data_dict['limit'] = 10_000
            self.test_dataset = self.get_dataset(conf, conf.data.test_path, split_name='test', **data_dict)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.training_dataset,
                          batch_size=self.conf.data.batch_size,
                          num_workers=self.conf.data.num_workers,
                          collate_fn=self.training_data_collector,
                          shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dev_dataset, batch_size=self.conf.data.test_batch_size,
                          num_workers=self.conf.data.num_workers,
                          collate_fn=self.training_data_collector)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.conf.data.test_batch_size,
                          num_workers=self.conf.data.num_workers,
                          collate_fn=self.validation_data_collector)

    def generation_dataloader(self, *args, **kwargs):
        test_dataset = self.get_dataset(self.conf, self.conf.data.test_path, split_name='test', limit=10_000,
                                        for_generation=True)
        return DataLoader(test_dataset, batch_size=self.conf.data.batch_size, num_workers=self.conf.data.num_workers,
                          collate_fn=self.generation_data_collector)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx=None) -> Any:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            if isinstance(v, dict):
                batch[k] = self.transfer_batch_to_device(v, device)
        return batch
