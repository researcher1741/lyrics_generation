import jsonlines
import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from tqdm import tqdm


class SimpleDataCollator():
    def __init__(self, pad_token_id) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        input_ids, labels, keys, schemas, languages = zip(
            *[(e['input_ids'], e['labels'], e['key'], e['schema'], e.get('language', 'english')) for e in examples])
        input_max_len = max([len(x) for x in input_ids])
        label_max_len = max([len(x) for x in labels])
        padded_input = []
        padded_output = []
        for in_ids, out_ids in zip(input_ids, labels):
            padded_input.append(torch.LongTensor(in_ids + [self.pad_token_id] * (input_max_len - len(in_ids))))
            padded_output.append(torch.LongTensor(out_ids + [-100] * (label_max_len - len(out_ids))))
        padded_input = torch.stack(padded_input, 0)
        padded_output = torch.stack(padded_output, 0)
        return {'input_ids': padded_input, 'labels': padded_output, 'keys': keys, 'schema': schemas,
                'languages': languages}


class SimpleDataset():
    def __init__(self, path, tokenizer, max_len, dataset_name, split_name, version, language) -> None:
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_len = max_len
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.version = version
        self.language = language
        self.examples = self._load_examples(path)

    def _load_examples(self, path):
        examples = list()
        batch_size = 512
        keys_batch = []
        input_batch = []
        output_batch = []
        schema_batch = []
        lang_batch = []
        with jsonlines.open(path) as lines:
            for line in tqdm(lines, desc='loading ' + self.split_name):
                input_str = line['input']
                output_str = line['output']
                key = line['key']
                schema = line['schema']
                lang = line['language']
                lang_batch.append(lang)
                keys_batch.append(key)
                schema_batch.append(schema)
                input_batch.append(input_str)
                output_batch.append(output_str)
                if len(input_batch) == batch_size:
                    input_ids = self.tokenizer(input_batch, max_length=self.max_len, truncation=True).input_ids
                    labels = self.tokenizer(output_batch, max_length=self.max_len, truncation=True).input_ids
                    for k, ids, l, s, la in zip(keys_batch, input_ids, labels, schema_batch, lang_batch):
                        examples.append({'key': k, 'input_ids': ids, 'labels': l, 'schema': s, 'language': la})
                    input_batch = []
                    output_batch = []
                    keys_batch = []
                    schema_batch = []
                    lang_batch = []
            if len(input_batch) > 0:
                input_ids = self.tokenizer(input_batch, max_length=self.max_len).input_ids
                labels = self.tokenizer(output_batch, max_length=self.max_len).input_ids
                for k, ids, l, s, la in zip(keys_batch, input_ids, labels, schema_batch, lang_batch):
                    examples.append({'key': k, 'input_ids': ids, 'labels': l, 'schema': s, 'language': la})
        return examples

    def __getitem__(self, idx: int):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)
