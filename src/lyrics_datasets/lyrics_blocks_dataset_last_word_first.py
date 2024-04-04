from transformers.tokenization_utils import PreTrainedTokenizer
from src.lyrics_datasets.lyrics_blocks_dataset import LyricsBlockDataset
# from src.lyrics_generation_utils.constants import *
from src.lyrics_generation_utils.constants import *
from src.lyrics_generation_utils.utils import get_lyrics_tokenizer
import re


class LyricsBlockDatasetLastWordFirst(LyricsBlockDataset):
    def __init__(self, path_or_paths,
                 max_len,
                 tokenizer: PreTrainedTokenizer,
                 dataset_name,
                 split_name,
                 genre_mapping_path,
                 year_mapping_path,
                 language=None,
                 num_processes=1,
                 limit=-1,
                 rhyming_schema_dropout_probability=0.0,
                 decoder_start_token_id=None,
                 song_filter=None,
                 version='0.2.1',
                 for_decoder=False,
                 **kwargs
                 ) -> None:
        if for_decoder:
            decoder_start_token_id = None
        super().__init__(path_or_paths, max_len, tokenizer, dataset_name, split_name,
                         genre_mapping_path, year_mapping_path,
                         language, num_processes, limit=limit,
                         rhyming_schema_dropout_probability=rhyming_schema_dropout_probability,
                         decoder_start_token_id=decoder_start_token_id, song_filter=song_filter, version=version,
                         **kwargs)
        self.for_decoder = for_decoder

    def tokenize_lyrics(self, songs, lyrics_key, is_prompt):
        if is_prompt:
            return super().tokenize_lyrics(songs, lyrics_key, is_prompt)
        lyrics = songs[lyrics_key]
        new_lyrics = []
        for block in lyrics:
            new_block = []
            if block == None:
                new_lyrics.append([])
                continue
            start_sentence_idx = 0
            if len(block) > 0 and block[0][0].endswith(TAG_END):
                start_sentence_idx = 1
            for sentence in block[start_sentence_idx:]:
                if len(sentence) < 2:
                    new_block.append(sentence)
                    continue
                if sentence[-1] != SENTENCE_END:
                    sentence.append(SENTENCE_END)
                j = 2
                while j < len(sentence) + 1:
                    last_word = sentence[-j]
                    if last_word.endswith(')'):
                        j += 1
                        if last_word.startswith('('):
                            continue
                        while j < len(sentence) + 1 and not sentence[-j].startswith('('):
                            j += 1
                        j += 1
                        continue
                    last_word = re.sub(r'[!@#$%^&\*\(\)\-_\+=|\\}{\[\]:;"\'?><.,]', '',
                                       last_word).strip()  # strip off special symbols
                    if len(last_word) == 0:
                        j += 1
                        continue
                    else:
                        break
                sentence = [last_word, SEP] + sentence
                new_block.append(sentence)
            new_lyrics.append(new_block)

        lyrics = new_lyrics
        num_blocks = [len(x) for x in lyrics]
        flat_lyrics = [s for block in lyrics for s in block]
        flat_lyrics = [' '.join(s).replace(' ' + SENTENCE_END, SENTENCE_END) for s in flat_lyrics]

        flat_encoded_lyrics = self.tokenizer(flat_lyrics, add_special_tokens=False)

        encoded_lyrics = self.reverse_flat(flat_encoded_lyrics, num_blocks)
        return encoded_lyrics

    def __getitem__(self, index):
        ret = super().__getitem__(index)
        if not self.for_decoder:
            return ret
        # {
        #     'input_ids': input_ids,
        #     'labels': labels,
        #     'rhyme_token_ids': rhyme_token_ids,
        #     'relative_position_ids': relative_position_ids, 
        #     'clear_schema': clear_schema,
        #     'task_name': 'lyrics generation',
        #     'language': language
        # }
        if index==1:
            print("Examples of the input printed:")
            for k,v in ret.items():
                if 'ids' in k:
                    print(f"key: {k}, value: {v}")
                else:
                    print(f"key: {k}, value: {self.tokenizer.decode(v)}")
                    
        input_ids = ret['input_ids']
        labels = ret['labels']
        if self.tokenizer.eos_token is not None:
            input_ids = input_ids[:-1]

        aux = input_ids + self.tokenizer.encode(GENERATE, add_special_tokens=False) + labels
        if len(aux) > self.max_len:
            aux = aux[:self.max_len - 1] + [labels[-1]]
        ret['input_ids'] = aux
        ret['labels'] = aux
        ret['relative_position_ids'] = None
        ret['rhyme_token_ids'] = None
        return ret


import os

if __name__ == '__main__':
    path = './LG/DATA/genius_section_0.2/test.3500.jsonl'
    tokenizer = get_lyrics_tokenizer('gpt2-large')
    genres_mapping_path = 'data/genres_mapping_100.txt'
    dataset = LyricsBlockDatasetLastWordFirst(path, 300, tokenizer, 'genius', 'test', genres_mapping_path,
                                              for_decoder=True)
    os.system('reset')
    print(dataset[1])
    print(len(dataset))
