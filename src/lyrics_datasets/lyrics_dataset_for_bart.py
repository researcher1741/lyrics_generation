from typing import Any, Dict, List
from transformers import DefaultDataCollator
from transformers.tokenization_utils import PreTrainedTokenizer
from lyrics_datasets.lyrics_dataset import LyricsDataset
from lyrics_generation_utils.constants import *
from urllib.parse import unquote
import html
import torch
from lyrics_generation_utils.utils import get_lyrics_tokenizer
from torch.nn.utils.rnn import pad_sequence


class BARTDataCollector(DefaultDataCollator):
    def __init__(self, tokenizer):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        input_ids = [torch.LongTensor(f['input_ids']) for f in features]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        labels = None
        if 'labels' in features[0]:
            labels = [torch.LongTensor(f['labels']) for f in features]
            labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_id)
        attention_mask = input_ids != self.pad_id
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}


class BARTLyricsDataset(LyricsDataset):
    def __init__(self, path_or_paths, max_len, tokenizer: PreTrainedTokenizer,
                 dataset_name, split_name, genre_mapping_path, language=None, num_processes=1,
                 prompt_component_probability=0.5, limit=-1, with_prompt=True) -> None:
        self.continue_token_id = tokenizer(CONTINUE_TOKEN, add_special_tokens=False)['input_ids']
        self.lyrics_token_id = tokenizer(LYRICS, add_special_tokens=False)['input_ids']
        self.generated_token_id = tokenizer(GENERATED, add_special_tokens=False)['input_ids']
        self.end_lyrics_token_id = tokenizer(END_LYRICS, add_special_tokens=False)['input_ids']

        super().__init__(path_or_paths, max_len, tokenizer, dataset_name, split_name, genre_mapping_path, language,
                         num_processes, prompt_component_probability, limit, with_prompt)

    def _batch_tokenise_lyrics(self, songs):
        artists = [ARTIST + html.unescape(unquote(a)) for a in songs['artist']]
        titles = [TITLE + html.unescape(unquote(t.replace('_', ' '))) for t in songs['title']]
        lyrics = [html.unescape(unquote(s if isinstance(s, str) else '\n'.join(s))) + ' ' + END_LYRICS for s in
                  songs['lyrics']]
        if 'emotion_tags' in songs:
            emotion_tags = [list({et for et in ets}) if ets is not None and len(ets) > 0 else '' for ets in
                            songs['emotion_tags']]  # list of lists deal with it properly
        else:
            emotion_tags = None
        topics = self.get_topics(songs)
        genres = self.get_genres(songs)
        # langs = [LANG + l.split(':')[0] if l is not None else 'English' for l in songs['lang']]
        input_ids, labels = self.buil_input_ids_and_labels(artists, titles, lyrics, genres, topics, emotion_tags)

        return {'input_ids': input_ids, 'labels': labels, 'lyrics_lengths': [len(x) for x in input_ids]}

    def get_genres(self, songs):
        aux = []
        if 'genre' in songs or 'genres' in songs:
            for genres in songs.get('genre', songs.get('genres')):
                if isinstance(genres, list):
                    if len(genres) == 1:
                        genres = genres[
                            0]  ## TODO refine the automatic extraction of genres by counting the categories.
                    else:
                        genres = ''
                if genres is not None and genres != '':
                    genres = html.unescape(unquote(genres))
                    genres = self.genre_mapping.get(genres, genres)
                    aux.append(GENRE + genres)
                else:
                    aux.append('')
            genres = aux
        else:
            genres = None
        return genres

    def get_topics(self, songs):
        if 'topics' in songs:
            ## TODO should improve filtering with POS tags and offensive word vocabulary
            topics = []
            for t in songs['topics']:
                if t is not None:
                    aux = [x.lower() for x in set(t) if
                           len(x) > 3 and x not in {'does', 'doesn', 'didn', 'would', 'nigga', 'niggas', 'wouldn',
                                                    'could', 'couldn'}]
                else:
                    aux = []
                topics.append(aux)
        else:
            topics = None
        return topics

    def build_prompt(self, artist, title, genre, emotions, topics, bos_id):
        i_prompt = artist + title
        if len(genre) > 0:
            i_prompt += genre
        if len(emotions) > 0:
            i_prompt += self.sample_from_collection(emotions, self.encoded_emotion_tag, 2)
        if len(topics) > 0:
            i_prompt += self.sample_from_collection(topics, self.encoded_topics_tag, 3)
        i_prompt = [bos_id] + i_prompt
        return i_prompt

    def build_label(self, lyrics):
        label = lyrics[:self.max_len - 1]
        if len(lyrics) > self.max_len:
            label += self.continue_token_id
        elif len(lyrics) == self.max_len:
            label += self.end_lyrics_token_id
        else:  # len(ly) < self.max_len
            label = lyrics  # end_lyrics_token_id already included
        return label

    def append_song_chunks(self, max_input_size, lyrics, current_prompt, input_ids, labels):
        i = max_input_size
        chunk_count = 1
        while i < len(lyrics):
            label = lyrics[i:i + max_input_size]
            if i + max_input_size < len(lyrics):
                label += self.continue_token_id
            if len(label) < 5:
                prev_label = labels[-1]
                if len(prev_label) < self.max_len:
                    labels[-1] = prev_label + self.end_lyrics_token_id
                else:
                    labels[-1] = prev_label[:-1] + self.end_lyrics_token_id
                break
            generated_i = self.tokenizer(GENERATED + str(chunk_count), add_special_tokens=False)['input_ids']
            # subtract 2 to account for lyrics_token_id and continue_token_id
            new_max_prompt_len = self.max_len - len(current_prompt) - len(generated_i) - 2
            in_text = current_prompt + generated_i + self.lyrics_token_id + lyrics[
                                                                            i - new_max_prompt_len: i] + self.continue_token_id
            if len(in_text) > self.max_len:
                self.logger.warning("input text exceeding max len, this should not happen!")
            input_ids.append(in_text)
            labels.append(label)
            i += new_max_prompt_len
            chunk_count += 1

    def buil_input_ids_and_labels(self, artists, titles, lyrics, genres, topics, emotions):
        encoded_artists = self.tokenizer(artists, add_special_tokens=False)['input_ids']
        encoded_titles = self.tokenizer(titles, add_special_tokens=False)['input_ids']
        encoded_lyrics = self.tokenizer(lyrics, add_special_tokens=False)['input_ids']
        encoded_genres = self.tokenizer(genres, add_special_tokens=False)['input_ids'] if genres else None
        encoded_topics = [self.tokenizer(t, add_special_tokens=False)['input_ids'] if len(t) > 0 else [] for t in
                          topics] if topics else None
        encoded_emotions = [self.tokenizer(e, add_special_tokens=False)['input_ids'] if len(e) > 0 else [] for e in
                            emotions] if emotions else None
        generated_0 = self.tokenizer(GENERATED + '0', add_special_tokens=False)['input_ids']

        input_ids = []
        labels = []
        bos_id = self.tokenizer.bos_token_id
        for a, t, g, e, top, ly in zip(encoded_artists, encoded_titles, encoded_genres, encoded_emotions,
                                       encoded_topics, encoded_lyrics):
            i_prompt = self.build_prompt(a, t, g, e, top, bos_id)
            i_prompt_len = len(i_prompt) + len(generated_0)

            if i_prompt_len > self.max_len:
                self.logger.warning(
                    f'input prompt already exceed maximum input len: {i_prompt_len}, {self.max_len}. Skipping this example')
                continue
            label = self.build_label(ly)

            input_ids.append(i_prompt + generated_0 + self.lyrics_token_id)
            labels.append(label)

            max_input_size = self.max_len - len(i_prompt) - len(generated_0) - 1
            self.append_song_chunks(max_input_size, ly, i_prompt, input_ids, labels)
        return input_ids, labels

    def __getitem__(self, index):
        example = self.examples[index]
        return {'input_ids': example['input_ids'], 'labels': example['labels']}


from torch.utils.data import DataLoader

if __name__ == '__main__':
    path = 'data/genius/train.jsonl'
    tokenizer = get_lyrics_tokenizer('facebook/bart-base')
    max_len = 256
    dataset = BARTLyricsDataset(path, max_len, tokenizer, 'genius', 'test', 'data/genres_mapping_100.txt')
    data_collectors = BARTDataCollector(tokenizer)
    loader = DataLoader(dataset, batch_size=16, num_workers=0,
                        collate_fn=data_collectors)

    for batch in loader:
        print(batch)
        print()
