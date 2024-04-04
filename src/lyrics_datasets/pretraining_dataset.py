from typing import Any, Dict
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorMixin
import torch


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
        labels = padded_input_ids.clone()
        labels[labels == self.pad_token] = -100
        return {'input_ids': padded_input_ids, 'labels': labels}


class PretrainingDataset(Dataset):
    def __init__(self, path, max_len, dataset_name,
                 split_name, version,
                 lyrics_token_id,
                 num_processes=16,
                 limit=-1,
                 for_generation=False,
                 **kwargs) -> None:
        self.path = path
        self.max_len = max_len
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.version = version
        self.num_processes = num_processes
        self.limit = limit
        self.columns_to_remove = ['title', 'artist', 'lang', 'lyrics', 'source', 'emotions', 'genre', 'is_explicit',
                                  'topics', 'summary', 'input_str', 'prompt']
        self.for_generation = for_generation
        self.lyrics_token_id = lyrics_token_id
        self.examples = self.__load_data()

    def __load_data(self):
        print(f'{self.dataset_name}-{self.split_name}-v{self.version}')
        dataset = load_dataset("json", name=f'{self.dataset_name}-{self.split_name}-v{self.version}',
                               data_files=self.path, keep_in_memory=False)['train']
        aux = list(set(self.columns_to_remove) & set(dataset.column_names))
        dataset = dataset.remove_columns(aux)
        if 'token_len' in dataset.column_names:
            dataset = dataset.filter(lambda elem: elem['token_len'] < self.max_len)
        if 0 < self.limit < len(dataset):
            dataset = dataset.select(range(self.limit))
        if self.for_generation:
            def get_prompt(elem):
                input_ids = elem['input_ids']
                input_ids = input_ids[:input_ids.index(self.lyrics_token_id) + 1]
                elem['input_ids'] = input_ids
                return elem

            dataset = dataset.map(get_prompt)
        print(f"Input example\n")
        for k,v in dataset[0].items():
            if 'ids' in k:
                print("{k}: {v}")
            else:
                print("{k}: {v}")
        return dataset

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]['input_ids']
