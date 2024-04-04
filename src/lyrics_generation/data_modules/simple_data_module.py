from typing import Any
import pytorch_lightning as pl

from src.lyrics_datasets.simple_dataset import SimpleDataset
from src.lyrics_datasets.simple_dataset import SimpleDataCollator
import torch
from src.lyrics_generation_utils.utils import get_info_logger
from torch.utils.data import DataLoader


class SimpleDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, conf, data_collator=None):
        super().__init__()
        self.conf = conf
        self.data_version = conf.data.version
        self.tokenizer = tokenizer
        self.logger = get_info_logger(__name__)
        self.tasks = conf.train.tasks
        self.data_collator = data_collator
        self.training_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        if data_collator is None:
            self.data_collator = SimpleDataCollator(tokenizer.pad_token_id)

    def get_dataset(self, paths, split_name, **kwargs):
        self.logger.info(f'{split_name.upper()} Data: {paths}')
        return SimpleDataset(paths, self.tokenizer, self.conf.data.max_len, 'simple_dataset',
                             split_name, self.data_version, self.conf.data.language)

    def setup(self, *args, **kwargs):

        split_to_setup = ['train', 'test', 'dev']
        if 'split_to_setup' in kwargs:
            split_to_setup = kwargs['split_to_setup']
        self.logger.info('Preparing datasets')

        if 'train' in split_to_setup and self.training_dataset is None:
            self.training_dataset = self.get_dataset(self.conf.data.train_path, 'train')
            print(f"####################    self.training_dataset {len(self.training_dataset)}    ##################")
        if 'dev' in split_to_setup and self.dev_dataset is None:
            self.dev_dataset = self.get_dataset(self.conf.data.validation_path, 'dev')
        if 'test' in split_to_setup and self.test_dataset is None:
            self.test_dataset = self.get_dataset(self.conf.data.test_path, 'test')

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.training_dataset,
                          batch_size=self.conf.data.batch_size,
                          num_workers=0,  # self.conf.data.num_workers,
                          collate_fn=self.data_collator,
                          shuffle=True)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.dev_dataset,
                          batch_size=self.conf.data.test_batch_size,
                          num_workers=0,  # self.conf.data.num_workers,
                          collate_fn=self.data_collator,
                          shuffle=False)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.test_dataset,
                          batch_size=self.conf.data.test_batch_size,
                          num_workers=0,  # self.conf.data.num_workers,
                          collate_fn=self.data_collator,
                          shuffle=False)

    def generation_dataloader(self, task=None):
        return DataLoader(self.test_dataset,
                          batch_size=self.conf.data.test_batch_size,
                          num_workers=0,
                          collate_fn=self.data_collator,
                          shuffle=False)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx=None) -> Any:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            if isinstance(v, dict):
                batch[k] = self.transfer_batch_to_device(v, device)
        return batch
