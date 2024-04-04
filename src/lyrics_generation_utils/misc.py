from transformers import AutoTokenizer
from tqdm import tqdm
import os
import jsonlines
import re
from sklearn.utils import shuffle
from src.lyrics_generation_utils.utils import get_info_logger

CHORUS_RE = r"[\[(]*chorus[:\]) x0-9]*.*(\n|$)"
CHORUS_TOKEN = '<chorus>'
logger = get_info_logger()


def build_chorus_dataset(songs):
    logger.info('Building chorus dataset...')
    not_found = 0
    end_chorus_not_found = 0
    # songs_with_chorus = list()
    for song in tqdm(songs, desc='filtering songs and matching choruses'):
        # song = dict(song)
        if song['title'] == 'How_Many_Times':
            print()

        lyrics = song['lyrics']
        if isinstance(lyrics, list):
            lyrics = '\n'.join(lyrics)
        lyrics = lyrics.strip()
        if 'chorus' not in lyrics.lower():
            continue
        lyrics = re.sub(r'\n\n\n+', '\n\n', lyrics)
        chorus_matches = list(re.finditer(CHORUS_RE, lyrics, re.MULTILINE | re.IGNORECASE))

        if len(chorus_matches) == 0:
            not_found += 1
            continue

        choruses = list()
        for match in chorus_matches:
            start, end = match.span()
            end_chorus_idx = list(re.finditer(r'(\n\n+|[\[({]verse)|\Z', lyrics[start:].strip(), re.IGNORECASE))
            if len(end_chorus_idx) == 0 and end < len(lyrics) - 1:
                if len(choruses) > 0:
                    char_len = len(choruses[0])
                    end_chorus_idx = start + char_len
                    choruses.append((start, end, end_chorus_idx))
                else:
                    end_chorus_not_found += 1
                continue
            if end >= len(lyrics) - 1:
                end_chorus_idx = end
            else:
                end_chorus_idx = end_chorus_idx[0].span()[0] + start

            choruses.append((start, end, end_chorus_idx))
        if len(choruses) > 0:
            first_chorus = choruses[0]
            first_chorus = lyrics[first_chorus[0]:first_chorus[-1]]
            first_chorus = re.sub(CHORUS_RE, '', first_chorus)
            if len(first_chorus) < 10:
                continue
        song['chorus_indices'] = choruses
        song['lyrics'] = lyrics
        # if len(choruses) > 1:
        #     songs_with_chorus.append(song)
    logger.info('Done Filtering & Matching')
    return songs


def build_structure_prompt(songs):
    for s in songs:
        if len(s['chorus_indices']) == 0:
            continue
        lyrics = s['lyrics']
        structure = []
        chorus_indices = s['chorus_indices']
        for i in range(len(chorus_indices)):
            start, _, _ = chorus_indices[i]
            if i > 0:
                _, _, prev_end = chorus_indices[i - 1]
            else:
                prev_end = 0
            verses = [x for x in lyrics[prev_end:start].split('\n\n') if len(x) > 0]
            structure += ['V'] * len(verses)
            structure += ['C']
        _, _, end = chorus_indices[-1]
        verses = [x for x in lyrics[end:].split('\n\n') if len(x) > 0]
        structure += ['V'] * len(verses)
        s['structure'] = structure
    return songs


def normalize_chorus_tag(songs):
    # new_songs = list()
    logger.info("Normalising chorus tags")
    for s in tqdm(songs, desc='Normalising'):
        # s = dict(s)
        if 'chorus_indices' not in s:
            s['chorus_indices'] = []
            continue
        lyrics = s['lyrics']
        matches = list(re.finditer(CHORUS_RE, lyrics, flags=re.IGNORECASE | re.MULTILINE))
        for m in reversed(matches):
            start, end = m.span()
            line_end = start + len(lyrics[start:].split('\n')[0])
            pre_chorus_token = '\n'
            if lyrics[start - 2:start] == '\n\n':
                pre_chorus_token = ''
            lyrics = lyrics[:start] + pre_chorus_token + CHORUS_TOKEN + '\n' + lyrics[line_end:].strip()

        s['lyrics'] = lyrics
        matches = re.finditer(fr'{CHORUS_TOKEN}', lyrics, flags=re.IGNORECASE | re.MULTILINE)
        spans = [m.span() for m in matches]
        enriched_spans = list()
        for (start, end, end_chorus), new_span in zip(s['chorus_indices'], spans):
            chorus_len = end_chorus - end
            aux = (new_span[0], new_span[1], new_span[1] + chorus_len + 1)
            enriched_spans.append(aux)
        aux = list()
        for st, _, ce in enriched_spans:
            x = lyrics[st:ce].replace(CHORUS_TOKEN, '').strip()
            aux.append(x)
        if all(len(x) == 0 for x in aux):
            continue
        s['chorus_indices'] = enriched_spans
        # new_songs.append(s)
    logger.info('Done normalising')
    return songs


def merge_datasets_and_normalise_chorus(wasabi_path, zh_path, outpath, train_perc=0.9, dev_perc=0.05):
    all_songs = dict()
    for f in ['train', 'dev', 'test']:
        full_path = os.path.join(wasabi_path, f + '.jsonl')
        with jsonlines.open(full_path) as lines:
            count = 0
            for song in tqdm(lines, desc=f'reading wasabi {f}'):
                if song['language'] != 'english':
                    continue
                count += 1
                all_songs[(song['title'].lower().replace('_', ' '), song['artist'].lower())] = song
                if count == 10000:
                    break

    for f in ['train', 'dev', 'test']:
        full_path = os.path.join(zh_path, f + '.jsonl')
        with jsonlines.open(full_path) as lines:
            count = 0
            for song in tqdm(lines, desc=f'reading zh data {f}'):
                if not song['lang'].startswith('en:'):
                    continue
                key = (song['title'].lower().replace('_', ' '), song['artist'].lower())
                if key not in all_songs:
                    all_songs['key'] = song
                count += 1
                if count == 10000:
                    break

    all_songs = list(all_songs.values())
    all_songs = build_chorus_dataset(all_songs)
    all_songs = normalize_chorus_tag(all_songs)
    all_songs = build_structure_prompt(all_songs)
    shuffle(all_songs)

    train_len = int(train_perc * len(all_songs))
    dev_len = int(dev_perc * len(all_songs))
    train_songs = all_songs[:train_len]
    dev_songs = all_songs[train_len:train_len + dev_len]
    test_songs = all_songs[train_len + dev_len:]
    for f, data in zip(['train', 'dev', 'test'], [train_songs, dev_songs, test_songs]):
        with jsonlines.open(os.path.join(outpath, f), 'w') as writer:
            for s in tqdm(data, desc=f'dumping {f}'):
                writer.write(s)


if __name__ == '__main__':
    # wasabi_path = 'data/wasabi'
    # zh_path = 'data/zh_music'
    # outpath = 'data/merged_dataset/'
    # merge_datasets_and_normalise_chorus(wasabi_path, zh_path, outpath)
    tokenizer = AutoTokenizer.from_pretrained('data/wasabi/wasabi_tokenizer')
    print()
