from typing import Any, List, Union
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from src.lyrics_datasets.lyrics_blocks_dataset import LyricsBlockDataset, MultitaskDataCollator
from src.lyrics_datasets.lyrics_blocks_dataset_last_word_first import LyricsBlockDatasetLastWordFirst
from src.lyrics_datasets.lyrics_dataset import LyricsDataset
from src.lyrics_datasets.pretraining_dataset_encoder_decoder import PretrainingDatasetEncoderDecoder, \
    PretrainingDatasetEncoderDecoderLastWordFirst
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader
from src.lyrics_datasets.rhyming_dataset import EnLastSyllablePronounceDataset, EnRhymesDataset

from src.lyrics_generation_utils.utils import get_info_logger, get_lyrics_tokenizer
from hydra import initialize, compose


class MultiDatasetDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, conf: DictConfig, decoder_start_token_id, data_collator=None):
        super().__init__()
        self.conf = conf
        self.data_version = conf.data.version
        self.tokenizer = tokenizer
        self.logger = get_info_logger(__name__)
        tok = get_lyrics_tokenizer(self.conf)
        tok.padding_side = 'left'
        self.tasks = conf.train.tasks
        data_collator = data_collator
        if data_collator is None:
            data_collator = MultitaskDataCollator(tokenizer, mlm=False)
        self.training_data_collector = data_collator
        self.validation_data_collector = data_collator
        self.decoder_start_token_id = decoder_start_token_id

    def _get_dataset_by_version(self):
        if str(self.data_version) == '0.1':
            return LyricsDataset
        elif str(self.data_version) == '0.2':
            return LyricsBlockDataset
        elif str(self.data_version) == '0.2.1':  # ex 0.3
            return LyricsBlockDatasetLastWordFirst
        elif str(self.data_version) == '0.2.2':
            return PretrainingDatasetEncoderDecoderLastWordFirst
        elif str(self.data_version) == '0.2.3':
            return PretrainingDatasetEncoderDecoder
        else:
            raise RuntimeError(f"Version {self.data_version} not supported.")

    def get_dataset(self, conf, paths, split_name, tokenizer, limit, **kwargs):
        self.logger.info(f'{split_name.upper()} Data: {paths}')
        dataset_constructor = self._get_dataset_by_version()
        return dataset_constructor(paths,
                                   conf.data.max_len, tokenizer,
                                   language=conf.data.language if 'language' in conf.data else None,
                                   dataset_name=conf.data.dataset_name,
                                   split_name=split_name,
                                   genre_mapping_path=conf.data.genre_mapping_path if 'genre_mapping_path'
                                                                                      in conf.data else None,
                                   year_mapping_path=conf.data.year_mapping_path if 'year_mapping_path'
                                                                                    in conf.data else None,
                                   limit=limit,
                                   rhyming_schema_dropout_probability=conf.data.rhyming_schema_dropout_probability if
                                   'rhyming_schema_dropout_probability' in conf.data else None,
                                   decoder_start_token_id=self.decoder_start_token_id,
                                   for_decoder=conf.data.for_decoder if 'for_decoder' in conf.data else False,
                                   add_language=conf.data.add_language if 'add_language' in conf.data else False)

    def setup(self, *args, **kwargs):
        split_to_setup = ['train', 'test', 'dev']
        if 'split_to_setup' in kwargs:
            split_to_setup = kwargs['split_to_setup']
        self.logger.info('Preparing datasets')
        conf = self.conf
        tokenizer = self.tokenizer
        data_dict = vars(conf.data)['_content']
        data_dict['limit'] = -1
        self.training_datasets, self.dev_datasets, self.test_datasets = {}, {}, {}
        training_dataset, dev_dataset, test_dataset = None, None, None
        for task in self.tasks:
            if task == 'lyrics_generation':
                if 'train' in split_to_setup:
                    training_dataset = self.get_dataset(conf, conf.data.train_path, split_name='train',
                                                        tokenizer=tokenizer,
                                                        **data_dict
                                                        )
                data_dict['limit'] = 10_000
                data_dict['prompt_component_probability'] = 1.0
                if 'dev' in split_to_setup:
                    dev_dataset = self.get_dataset(conf, conf.data.validation_path, split_name='dev',
                                                   tokenizer=tokenizer, **data_dict)
                if 'test' in split_to_setup:
                    test_dataset = self.get_dataset(conf, conf.data.test_path, split_name='test', tokenizer=tokenizer,
                                                    **data_dict)
            elif task == 'rhyming_phone':
                if 'train' in split_to_setup:
                    training_dataset = EnLastSyllablePronounceDataset(self.tokenizer, 'train',
                                                                      decoder_start_token_id=self.decoder_start_token_id)
                if 'dev' in split_to_setup:
                    dev_dataset = EnLastSyllablePronounceDataset(self.tokenizer, 'dev',
                                                                 decoder_start_token_id=self.decoder_start_token_id)
                if 'test' in split_to_setup:
                    test_dataset = EnLastSyllablePronounceDataset(self.tokenizer, 'test',
                                                                  decoder_start_token_id=self.decoder_start_token_id)

            elif task == 'rhyming_words':
                word_frequency_data_path = self.conf.data.word_frequency_data_path if hasattr(self.conf.data,
                                                                                              'word_frequency_data_path') else None
                if 'train' in split_to_setup:
                    training_dataset = EnRhymesDataset(self.tokenizer, 'train',
                                                       word_frequency_data_path=word_frequency_data_path,
                                                       decoder_start_token_id=self.decoder_start_token_id)
                if 'dev' in split_to_setup:
                    dev_dataset = EnRhymesDataset(self.tokenizer, 'dev',
                                                  word_frequency_counter=training_dataset.word_freq,
                                                  decoder_start_token_id=self.decoder_start_token_id)
                if 'test' in split_to_setup:
                    test_dataset = EnRhymesDataset(self.tokenizer, 'test',
                                                   word_frequency_counter=training_dataset.word_freq,
                                                   decoder_start_token_id=self.decoder_start_token_id)

            self.training_datasets[task] = training_dataset
            self.dev_datasets[task] = dev_dataset
            self.test_datasets[task] = test_dataset

    def __get_combined_dataloader(self, dataset_dict, is_train):
        ret = {}
        batch_size = self.conf.data.batch_size
        if not is_train:
            batch_size = self.conf.data.test_batch_size
        for task, dataset in dataset_dict.items():
            ret[task] = DataLoader(dataset,
                                   batch_size=batch_size,
                                   num_workers=self.conf.data.num_workers,
                                   collate_fn=self.training_data_collector,
                                   shuffle=False)
        return CombinedLoader(ret, mode='max_size_cycle')

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.__get_combined_dataloader(self.training_datasets, is_train=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.__get_combined_dataloader(self.dev_datasets, is_train=False)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.__get_combined_dataloader(self.test_datasets, is_train=False)

    def generation_dataloader(self, task=None):
        test_dataset = self.test_datasets
        if test_dataset is None:
            test_dataset = self.get_dataset(self.conf, self.conf.data.test_path, split_name='test',
                                            tokenizer=self.tokenizer, limit=10_000,
                                            prompt_component_probability=1.0)
        if task is not None and task in test_dataset:
            test_dataset = {task: test_dataset[task]}
        return self.__get_combined_dataloader(test_dataset, is_train=False)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx=None) -> Any:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            if isinstance(v, dict):
                batch[k] = self.transfer_batch_to_device(v, device)
        return batch


if __name__ == '__main__':
    CONFIG_NAME = 'root_t5'
    # MultiDatasetDataModule(tokenizer, conf, decoder_start_token_id = pl_module.model.config.decoder_start_token_id)
    with initialize(config_path="../../../conf", ):
        conf = compose(config_name=CONFIG_NAME)
    tokenizer = get_lyrics_tokenizer(conf)
    conf.data.test_batch_size = 8
    conf.data.batch_size = 8
    module = MultiDatasetDataModule(tokenizer, conf, decoder_start_token_id=tokenizer.bos_token_id)
    module.setup(split_to_setup={'test'})
    dataloader = module.test_dataloader()
    i = 0
    from tqdm import tqdm

    for batch in tqdm(dataloader):
        i += 1
