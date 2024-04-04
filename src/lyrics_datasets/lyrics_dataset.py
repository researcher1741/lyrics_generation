from os import remove
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import Features, concatenate_datasets, load_dataset
import numpy as np

from src.lyrics_generation_utils.constants import END_LYRICS
from src.lyrics_generation_utils.utils import get_info_logger, get_lyrics_tokenizer
from src.lyrics_generation_utils.constants import LYRICS_SPECIAL_TOKENS, EMOTIONS, ARTIST, TITLE, LYRICS, LANG, \
    GENRE, NUM_SYLLABLES
from urllib.parse import unquote
import html


class LyricsDataset:
    def __init__(self, path_or_paths, max_len, tokenizer: PreTrainedTokenizer,
                 dataset_name, split_name, genre_mapping_path,
                 year_mapping_path, language=None, num_processes=1,
                 prompt_component_probability=0.5, limit=-1, with_prompt=True,
                 version='0.1', **kwargs) -> None:
        super().__init__()
        if language is not None:
            self.language = language.lower()
        else:
            self.language = language
        self.split_name = split_name
        self.limit = limit
        self.with_prompt = with_prompt
        self.logger = get_info_logger(__name__)
        diff = set(LYRICS_SPECIAL_TOKENS) - set(tokenizer.get_vocab().keys()) - set(tokenizer.all_special_tokens)
        if len(diff) > 0:
            self.logger.warning(
                f"The passed tokenizer's vocabulary DOES NOT contain all LYRICS SPECIAL TOKENS.\n"
                f"Missing Tokens: {', '.join(diff)}.\n Continue at your own risk of suboptimal splitting.")
        self.dataset_name = dataset_name
        self.path_or_paths = path_or_paths
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.num_processes = num_processes
        self.pretrained_model_name = tokenizer.name_or_path
        if genre_mapping_path is not None:
            self.genre_mapping = self.load_genre_mapping(genre_mapping_path)
        else:
            self.genre_mapping = dict()
        if year_mapping_path is not None:
            self.year_mapping = self.load_genre_mapping(year_mapping_path)
        else:
            self.year_mapping = dict()
        self.prompt_component_probability = prompt_component_probability if split_name == 'train' else 1.0
        self.encoded_emotion_tag = self.tokenizer.encode(EMOTIONS, add_special_tokens=False)
        self.encoded_comma = self.tokenizer.encode(',', add_special_tokens=False)
        self.version = version
        self.types = set()
        self.examples = self.load_data()

    def load_genre_mapping(self, path):
        mapping = dict()
        with open(path) as lines:
            for line in lines:
                fields = line.strip().split('\t')
                mapping[fields[0]] = fields[1]
        return mapping

    def _batch_tokenise_lyrics(self, songs):
        # Cleaning
        conversor = {'title': 'encoded_title',
                     'artist': 'encoded_artist',
                     'mood': 'encoded_mood',
                     'scene': 'encoded_scene',
                     'lang': 'encoded_language',
                     'topics': 'encoded_topics',
                     'year': 'encoded_year',
                     'genre': 'encoded_genre',
                     'lyrics': 'encoded_lyrics',
                     'emotions': 'encoded_emotion_tags',
                     }
        print("hey")
        songs = dict((conversor[k], v) for (k, v) in songs.items() if k in conversor.keys())
        print("hoy")
        output = ['encoded_lyrics', 'encoded_title', 'encoded_artist', 'encoded_language', 'lyrics_lengths',
                  'encoded_genre', 'encoded_year', 'encoded_mood', 'encoded_emotion_tags', 'encoded_topics']
        output = dict((k, songs[k] if k in songs.keys() else None) for k in output)

        # HTML
        output['encoded_artist'] = [ARTIST + html.unescape(unquote(a)) for a in output['encoded_artist']]
        output['encoded_title'] = [TITLE + html.unescape(unquote(t.replace('_', ' '))) for t in output['encoded_title']]

        # lyrics
        print("###########", type(output['encoded_lyrics']))
        print("###########", [type(x) for x in output['encoded_lyrics']][:3])
        print("###########", output['encoded_lyrics'][0])
        if self.with_prompt:
            output['encoded_lyrics'] = [LYRICS + html.unescape(unquote(s if isinstance(s, str) else '\n'.join(s)))
                                        + ' ' + END_LYRICS for s in output['encoded_lyrics']]
        else:
            output['encoded_lyrics'] = [html.unescape(unquote(s if isinstance(s, str) else '\n'.join(s)))
                                        for s in output['encoded_lyrics']]
        #
        if output['encoded_emotion_tags']:
            output['encoded_emotion_tags'] = [list({et for et in ets}) if ets is not None and len(ets) > 0 else '' for
                                              ets in
                                              output['encoded_emotion_tags']]  # list of lists deal with it properly
        if output['encoded_topics']:
            topics, avoid = [], {'does', 'doesn', 'didn', 'would', 'nigga', 'niggas', 'wouldn', 'could', 'couldn'}
            for t in output['topics']:
                aux = [x.lower() for x in set(t) if len(x) > 3 and x not in avoid] if t is not None else []
                topics.append(aux)
            output['encoded_topics'] = topics

        output['encoded_language'] = [LANG + l.split(':')[0] if l is not None else 'English' for l in
                                      output['encoded_language']]

        for x in ['encoded_genre', 'encoded_year']:
            if output[x]:
                aux = []
                for genres in output.get(x):
                    if isinstance(genres, list):
                        genres = genres[0] if len(genres) == 1 else ''
                    if genres is not None and genres != '':
                        genres = html.unescape(unquote(genres))
                        # genres = self.genre_mapping.get(genres, genres)
                        aux.append(GENRE + genres)
                    else:
                        aux.append('')
                output[x] = aux
        for x in ['encoded_scene', 'encoded_mood']:
            if output[x] and isinstance(output[x], list):
                output[x] = ",".join(output[x])
            elif output[x]:
                output[x] = output[x]

        # Tokenization
        output['lyrics_lengths'] = [len(l) for l in output['encoded_lyrics']]
        output['encoded_lyrics'] = self.tokenizer(output['encoded_lyrics'],
                                                  add_special_tokens=False,
                                                  truncation=True,
                                                  max_length=self.max_len)['input_ids']

        for k in output.keys():
            output[k] = self.tokenizer(output[k], add_special_tokens=False)['input_ids'] if output[k] else None

        output['lyrics_lengths'] = [len(l) for l in output['encoded_lyrics']]
        for x in ['encoded_artist', 'encoded_titles', 'encoded_langs', 'encoded_genres', 'encoded_year']:
            output[x] = self.tokenizer(output[x], add_special_tokens=False)['input_ids']
        for x in ['encoded_emotion_tags', 'encoded_topics']:
            output[x] = [self.tokenizer(t, add_special_tokens=False)['input_ids'] if len(t) > 0 else [] for t in
                         output[x]] if output[x] else [[]]
        return {k: v for k, v in output.items() if v is not None}

    def _get_filtering_function(self):
        lang_condition = lambda x: x.get('lang') is None or x.get('lang', 'english').split(':')[
            0].lower() == self.language.lower()
        prompt_condition = lambda example: len(example['title']) != 0 or len(example['artist']) != 0 or len(
            example.get('genres', [])) != 0
        filtering_condition = None

        if self.language is not None and self.split_name != 'train':
            filtering_condition = lambda example: lang_condition(example) and prompt_condition(example)
        elif self.language is not None and self.split_name == 'train':
            filtering_condition = lang_condition
        elif self.language is None and self.split_name != 'train':
            filtering_condition = prompt_condition
        return filtering_condition

    def load_data(self):
        if isinstance(self.path_or_paths, str):
            dataset = load_dataset("json", name=f'{self.dataset_name}-{self.split_name}',
                                   data_files=self.path_or_paths, keep_in_memory=False)['train']
        else:
            datasets = []
            for path in self.path_or_paths:
                dataset = load_dataset('json', data_files=path)['train']
                if isinstance(dataset[0]['lyrics'], list):
                    def _map(song):
                        song['lyrics'] = '\n'.join(song['lyrics'])
                        return song

                    dataset = dataset.map(_map)
                datasets.append(dataset)
            dataset = concatenate_datasets(datasets)

        # Key correction
        conv2 = {'artist_name': 'artist',
                 'artists': 'artist',
                 'chronological_tag': 'year',
                 'emotion_tags': 'emotions',
                 'genre_tag': 'genre',
                 'genres': 'genre',
                 'langs': 'lang',
                 'mood_tag': 'mood',
                 'scene_tag': 'scene',
                 'song_name': 'title',
                 'topic': 'topics'}
        for k in dataset.features.keys():
            if k in conv2.keys():
                dataset = dataset.rename_column(k, conv2[k])
        if 'topics' not in dataset.features.keys():
            new_column = [[]] * len(dataset[list(dataset.features.keys())[0]])
            dataset = dataset.add_column("topics", new_column)
        if 'emotions' not in dataset.features.keys():
            new_column = [[]] * len(dataset[list(dataset.features.keys())[0]])
            dataset = dataset.add_column("emotions", new_column)

        #   dataset MAPPING
        filter_fun = self._get_filtering_function()
        if filter_fun is not None:
            dataset = dataset.filter(filter_fun, num_proc=8)
        if 0 < self.limit < len(dataset):
            dataset = dataset.select(range(self.limit))
        dataset = dataset.map(self._batch_tokenise_lyrics,
                              batched=True,
                              batch_size=2048,
                              remove_columns=dataset.column_names,
                              num_proc=8)
        # dataset = dataset.sort('lyrics_lengths', reverse=True)
        idx = sum([int(x > self.max_len) for x in dataset['lyrics_lengths']])
        self.logger.info(f'{idx + 1} songs over {len(dataset)} will be truncated')
        print(f"\n {dataset.features}\n")
        return dataset

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

    def __getitem__(self, index):
        example = self.examples[index]
        lyrics, artist, title, language = example['encoded_lyrics'], \
                                          example['encoded_artist'], \
                                          example['encoded_title'], \
                                          example['encoded_language']
        if not self.with_prompt:
            return {'input_ids': lyrics, 'prompt_len': 0}
        genres = example['encoded_genre'] if 'encoded_genre' in example else None
        emotions = example['encoded_emotion_tags'] if 'encoded_emotion_tags' in example else None
        topics = example['encoded_topics'] if 'encoded_topics' in example else None

        if self.prompt_component_probability < 1.0:
            probs = np.random.uniform(0, 1, 6)
        else:
            probs = np.ones(6)
        emotions = None if emotions is None or len(emotions) == 0 else emotions
        emotions = self.sample_from_collection(emotions, self.encoded_emotion_tag, 2)
        topics = self.sample_from_collection(topics, self.encoded_topics_tag, 3)
        inp = []
        for p, data in zip(probs, [artist, title, language, genres, emotions, topics]):
            if p < self.prompt_component_probability or data is None or len(data) == 0:
                continue
            inp.extend(data)
        prompt_len = len(inp) + 1  # + 1 for the <LYRICS> token
        inp.extend(lyrics)
        if len(inp) > self.max_len:
            inp = inp[:self.max_len]
            if inp[-1] != self.tokenizer.convert_tokens_to_ids([END_LYRICS])[0]:
                inp[-1] = self.tokenizer.convert_tokens_to_ids([END_LYRICS])[0]
        return {'input_ids': inp, 'prompt_len': prompt_len}

    def __len__(self):
        return len(self.examples)


if __name__ == '__main__':
    path = 'data/wasabi/test.jsonl'
    max_len = 1024
    tokenizer = get_lyrics_tokenizer('google/t5-v1_1-large')
    dataset = LyricsDataset(path, max_len, tokenizer, 'wasabi', 'test', 'data/genres_mapping_100.txt')
    dataset[0]
    print()
