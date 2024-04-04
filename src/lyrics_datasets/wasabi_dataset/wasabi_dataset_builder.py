import multiprocessing
import time
import os
import json
from multiprocessing import Queue
from tqdm import tqdm 
import logging
import sys
import jsonlines
from multiprocessing import Manager, Process
from collections import Counter, defaultdict
import datetime
import re
from urllib.parse import unquote
import html
import math
import numpy as np

PRECHORUS_RE = r"[\[( ]pre[\[( ]*chorus[:\]) x0-9]*.*(\n|$)"
CHORUS_RE = r"[^pre][the \[(repeat]*chorus[:\]) x0-9]*.*(\n|$)"
VERSE_RE = r"^[\[(]*verse[:\]) 0-9]*\n"
SOLO_RE = r"[\[(< ]solo[ :\])>](\n|$)"
BRIDGE_RE = r"[the \[(]*bridge[:\]) x0-9]*.*(\n|$)"

VERSE_TOKEN= '<verse>'
BRIDGE_TOKEN = '<bridge>'
CHORUS_TOKEN = '<chorus>'
PRECHORUS_TOKEN = "<prechorus>"
SOLO_TOKEN = '<solo>'

EMOTION_FILE_NAME = 'emotion-tags.json'
TOPIC_MODEL_FILE_NAME = 'topic-models.json'
SONG_TOPIC_FILE_NAME = 'song-topic.json'
SONG_FILE_NAME = 'song.json'
SOCIAL_TAG = 'social-tags.json'
ISO_LANGUAGE_MAPPING_FILE = 'iso_languages_map.txt'

logger = logging.getLogger('Wasabi Dataset Builder')
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def read_iso_language_mapping(wasabi_folder):
    short2long = dict()
    with open(os.path.join(wasabi_folder, ISO_LANGUAGE_MAPPING_FILE)) as lines:
        for line in lines:
            language_name, iso1, iso2, iso3, iso4 = line.strip().split('\t')
            short2long[iso1] = language_name
            short2long[iso2] = language_name
            short2long[iso3] = language_name
            short2long[iso4] = language_name
    return short2long

def normalise_language(language, short2long):
    return short2long.get(language, language).lower()

def read_social_tags(wasabi_folder):
    with open(os.path.join(wasabi_folder, SOCIAL_TAG)) as reader:
        social_data = json.load(reader)
    socials = dict()
    for d in social_data:
        oid = d.get('song_id', dict()).get('$oid', None)
        if oid is not None:
            socials[oid] = d
    return socials

def read_emotion_annotations(wasabi_folder):
    with open(os.path.join(wasabi_folder, EMOTION_FILE_NAME)) as reader:
        emotion_data = json.load(reader)        
    emotions = dict()
    for d in emotion_data:
        oid = d.get('song_id', dict()).get('$oid', None)
        if oid is not None:
            emotions[oid] = d
    return emotions

def lazy_read_songs(wasabi_folder, song_queue:Queue):
    reads = []
    with open(os.path.join(wasabi_folder, SONG_FILE_NAME)) as lines:
        for line in lines:
            line = line.strip()            
            if line == '[{': ## first line
                line = line[1:]
                reads.append(line)
            elif line == '},{': ## end objct
                reads.append('}')
                # logger.info(f'{os.getpid()}: adding song')
                song_queue.put(json.loads('\n'.join(reads)))
                reads = ['{']
            elif line == '}]': ## end of file
                reads.append('}')
                # logger.info(f'{os.getpid()}: adding song')
                song_queue.put(json.loads('\n'.join(reads)))
            else:
                reads.append(line)
    song_queue.put(None)

def maybed_report_progress(counter, start_time):
    if counter % 10000 == 0:
        current_time = time.time()
        current_time = datetime.datetime.now()
        elapsed = current_time - start_time
        # elapsed_str = str(datetime.timedelta(seconds=elapsed))
        logger.info(f'[Process {os.getpid()}] Read {counter} [{elapsed}]')
    
def read_topic_model(wasabi_folder):
    with open(os.path.join(wasabi_folder, TOPIC_MODEL_FILE_NAME)) as file:
        topic_models = json.load(file)
        topic_models = {t['topic_id']: t['terms'] for t in topic_models}
    with open(os.path.join(wasabi_folder, SONG_TOPIC_FILE_NAME)) as file:
        songs = json.load(file)
    ret = {}
    for s in songs:
        song_id = s['id_song']
        topics = Counter({t['topic']:t['probability'] for t in s['topics']})
        top_ranked_topic, _ = topics.most_common(1)[0]
        words = topic_models[top_ranked_topic]
        ret[song_id] = words
    return ret
        
def build_dataset(wasabi_folder, output_file):
    logger.info('Reading Emotion Annotations')
    emotions = read_emotion_annotations(wasabi_folder)
    logger.info('Reading Social Tags')
    socials = read_social_tags(wasabi_folder)
    short2long = read_iso_language_mapping(wasabi_folder)
    logger.info('Reading Topic Models')
    songid2topics = read_topic_model(wasabi_folder)
    song_queue = Manager().Queue(100000)
    song_queuer_process = Process(target=lazy_read_songs, args=[wasabi_folder, song_queue])
    song_queuer_process.start()
    logger.info('Reading Songs')
    counter = 0
    start_time = datetime.datetime.now()
    deduplicated = 0
    duplicates = 0
    dumped = dict()
    with jsonlines.open(output_file, 'w') as json_writer:
        while True:
            song = song_queue.get()
            if song is None:
                break
            maybed_report_progress(counter, start_time)
            counter += 1
            song_id = song['_id'].get('$oid', None)
            if song_id is None:
                continue
            lyrics = song.get('lyrics', None)
            if lyrics is None or len(lyrics.strip()) == 0:
                continue
            artist = html.unescape(unquote(song.get('name', ''))).strip()
            title = html.unescape(unquote(song["urlSong"])).split(':')[-1].strip()
            lyrics = html.unescape(unquote(lyrics)).replace('<br>', '\n').strip()
            if len(lyrics) == 0: ## This can still happen if lyrics has just <br> tokens.
                continue
            # lyrics_aux = re.sub('\n+', ' ', lyrics)
            if len(lyrics.split(' ')) < 20:
                continue
            if (artist, title) in dumped:
                if lyrics == dumped[(artist, title)]:
                    deduplicated += 1
                    continue
                else:
                    duplicates += 1
            album_title = html.unescape(unquote(song.get('albumTitle', ''))).strip()
            if album_title == '':
                album_title = None
            rank = int(song.get('rank', -1))
            bpm = str(song.get('bpm', -1.0)).strip()
            if isinstance(bpm, str) and len(bpm) == 0:
                bpm = -1.0
            else:
                bpm = float(bpm)
            pub_date = str(song.get("publicationDate", '').strip())
            language = str(song.get('language', '').strip())
            if language == '':
                language = str(song.get('language_detect', '').strip())
            language = normalise_language(language, short2long)
            topics = songid2topics.get(song_id, [])
            summary = '\n'.join(song.get('summary', ''))
            chords = song.get("chords_metadata", {})
            genre = html.unescape(unquote(song.get("album_genre", ''))).strip()
            explicit_lyrics = song["explicitLyrics"]
            emotion_tags = []
            emotion_tags_scores = []
            social_tags = []
            social_tags_scores = []
            if song.get('has_emotion_tags', '').lower() == 'true':
                emotion_tags = emotions.get(song_id, dict()).get('emotions', [])
                if emotion_tags != '':
                    emotion_tags_scores = [e['nbr_tags'] for e in emotion_tags]
                    emotion_tags = [html.unescape(unquote(e['emotion_tag'])).strip() for e in emotion_tags]
            if song.get('has_social_tags', '').lower() == 'true':
                social_tags = socials.get(song_id, dict()).get('socials', [])
                if social_tags != '':
                    social_tags_scores = [e['nbr_tags'] for e in social_tags]
                    social_tags = [html.unescape(unquote(e['social_tag'])).strip() for e in social_tags]
            json_writer.write(dict(title= title, album_title= album_title, rank= rank, bpm = bpm, artist = artist, 
                        pub_data= pub_date,  lang = language, chords = chords, genre = genre, lyrics=lyrics, 
                        explicit_lyrics=explicit_lyrics, 
                        emotion_tags = emotion_tags, emotion_tags_scores = emotion_tags_scores, 
                        social_tags=social_tags, social_tags_scores=social_tags_scores, 
                        topics = topics, summary=summary))
            dumped[(artist, title)] = lyrics
            
    song_queuer_process.join()
    deduplicated_again_counter = deduplicate_again(output_file)
    normalise_chorus_tags(output_file)

    logger.info('All Done;')
    logger.info(f'Duplicates: {duplicates}\nDeduplicated: {deduplicated}\nDeduplicated again: {deduplicated_again_counter}')
    logger.info('Peace!')


def normalise_chorus_tags(input_path):
    logger.info("Normalising chorus tags")
    with jsonlines.open(input_path) as reader, jsonlines.open(input_path + '.chorusnorm', 'w') as writer:
        for s in tqdm(reader, desc='Normalising'):
            lyrics = s['lyrics']
            lyrics = re.sub(PRECHORUS_RE, '\n\n' + PRECHORUS_TOKEN + '\n', lyrics, flags= re.IGNORECASE | re.MULTILINE)
            lyrics = re.sub(CHORUS_RE, '\n\n' + CHORUS_TOKEN + '\n', lyrics, flags= re.IGNORECASE | re.MULTILINE)
            lyrics = re.sub(SOLO_RE, SOLO_TOKEN + '\n', lyrics, flags= re.IGNORECASE | re.MULTILINE)
            lyrics = re.sub(BRIDGE_RE, '\n\n' + BRIDGE_TOKEN + '\n', lyrics, flags= re.IGNORECASE | re.MULTILINE)
            lyrics = re.sub(VERSE_RE, '\n\n' + VERSE_TOKEN + '\n', lyrics, flags= re.IGNORECASE | re.MULTILINE)
            lyrics = re.sub(CHORUS_TOKEN + '\n\n', CHORUS_TOKEN + '\n', lyrics, flags= re.IGNORECASE | re.MULTILINE, count=1)# replace first occurrence of chorus followed by 2 \n with one new line only 
            lyrics = re.sub('\n\n\n+', '\n\n', lyrics).strip()
            
            s['lyrics'] = lyrics
            writer.write(s)
        logger.info('Done normalising')

def check_equivalence(p1, p2):
    p1 = p1.lower()
    p2 = p2.lower()
    cond1 = p1 == p2
    p1_set = set(p1.split())
    p2_set = set(p2.split())
    cond2 = math.floor(len(p1_set - p2_set) / max(len(p1_set), 1) * 100) <= 30 # the difference in terms of words is less than 30%
    cond3 = math.floor(len(p2_set - p1_set) / max(len(p2_set), 1) * 100) <= 30
    # return int(p1 == p2 or p1 in p2 or p2 in p1)
    return int(cond1 or (cond2 and cond3))

def get_same_paragraph_matrix(paragraphs):
    same_paragraphs = np.zeros((len(paragraphs), len(paragraphs)))
    skip = set()
    for i in range(len(paragraphs)):
        if i in skip:
            continue
        p1 = paragraphs[i]
        for j in range(i+1, len(paragraphs)):
            p2 = paragraphs[j]
            are_same = check_equivalence(p1, p2)
            same_paragraphs[i][j] = are_same
            if are_same:
                skip.add(j)
    return same_paragraphs

def get_chorus_clusters(same_paragraphs):
    pairs = np.argwhere(same_paragraphs==1)
    choruses = defaultdict(list)
    for a, b in pairs:
        choruses[a].append(b)
    reverse_choruses = defaultdict(list)
    for k, l in choruses.items():
        reverse_choruses[tuple(l)].append(k)
    to_remove = set()
    for k, l in sorted(choruses.items(), key=lambda x: x[0]):
        if k in to_remove:
            continue
        other_k = set(reverse_choruses[tuple(l)])
        other_k.remove(k)
        for ok in other_k:
            to_remove.add(ok)
        choruses[k].extend(other_k)
    for x in to_remove:
        del choruses[x]

    if len(choruses) > 0:
        k = sorted(choruses.keys())[0]
        choruses = {k:choruses[k]}
    return choruses 

def build_textual_choruses(choruses, paragraphs):
    chorus_paragraph = dict()
    for ch_num, (k, l) in enumerate(choruses.items()):
        indices = sorted([k] + l)
        i = 0
        while i < len(indices):
            start = indices[i]
            j = i
            while j < len(indices) - 1 and indices[j+1] == indices[j] + 1:
                j += 1
            end = indices[j] + 1 # as indices[j] contains the index of the paragraph to be included
            chorus = CHORUS_TOKEN + '\n'  + '\n'.join(paragraphs[start:end])
            chorus_paragraph[start] = (chorus, start, end)
            i = j + 1
    return chorus_paragraph
            

def add_chorus_tags(input_path):
    logger.info('Adding chorus tags')
    choruses_added = 0
    tot = 0
    with jsonlines.open(input_path) as reader, jsonlines.open(input_path + '.chorusadded', 'w') as writer:
        bar = tqdm(reader, desc='adding choruses')
        for s in bar:
            lyrics = s['lyrics']
            if CHORUS_TOKEN in lyrics:
                # lyrics.replace(CHORUS_TOKEN, CHORUS_TOKEN[:-1] + f'{1}>')
                writer.write(s)
                continue
            tot += 1
            paragraphs = lyrics.split('\n\n')
            same_paragraphs_matrix = get_same_paragraph_matrix(paragraphs)
            choruses = get_chorus_clusters(same_paragraphs_matrix)
            if len(choruses) == 0:
                writer.write(s)
                continue
            choruses_added += 1
            chorus_paragraph = build_textual_choruses(choruses, paragraphs)
            new_paragraphs = []
            i = 0
            while i < len(paragraphs):
                if i in chorus_paragraph:
                    chorus, _, end = chorus_paragraph[i]
                    new_paragraphs.append(chorus)
                    i = end
                else:
                    new_paragraphs.append(paragraphs[i])
                    i += 1
            lyrics = '\n\n'.join(new_paragraphs)
            s['lyrics'] = lyrics
            writer.write(s)
            bar.set_postfix({'choruses_added': choruses_added, 'ratio': f'{choruses_added / tot:.2f}'})
            

def deduplicate_again(input_file):
    songs = dict()
    logger.info('Deduplicating Again')
    deduplicated = 0
    with jsonlines.open(input_file) as reader:
        for song1 in reader:
            artist = song1['artist']
            title = song1['title']
            key = (artist, title)
            if key in songs:
                song2 = songs[key]
                song2_lyrics = song2['lyrics']
                song1_lyrics = song1['lyrics']
                if len(song1_lyrics) > len(song2_lyrics):
                    songs[key] = song1
                    deduplicated += 1
            else:
                songs[key] = song1
    with jsonlines.open(input_file, 'w') as writer:
        for s in tqdm(songs.values(), desc='dumping deduplicated songs'):
            writer.write(s)
    return deduplicated


from argparse import ArgumentParser
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--wasabi_folder', type=str, default = 'data/wasabi/original/json/')
    parser.add_argument('--output_file', default = 'data/wasabi/songs.jsonl', type=str)
    args = parser.parse_args()

    # build_dataset(args.wasabi_folder, args.output_file)

    # normalise_chorus_tags(args.output_file)
    add_chorus_tags('data/wasabi/songs.jsonl.chorusnorm')