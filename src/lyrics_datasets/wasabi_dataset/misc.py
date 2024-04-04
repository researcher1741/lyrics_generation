from typing import List
import os
import jsonlines
from tqdm import tqdm

def extract_song_info(paths:List, out_file):
    with jsonlines.open(out_file, 'w') as writer:
        for f in paths:
            with jsonlines.open(f) as reader:
                for song in tqdm(reader):
                    genre = song['genre']
                    emotions = song['emotion_tags']
                    title = song['title']
                    artist = song['artist']
                    writer.write({'title':title, 'artist': artist, 'genre':genre, 'emotions':emotions})

if __name__ == '__main__':
    base_path = 'data/wasabi'
    paths = [os.path.join(base_path, 'train.jsonl'), os.path.join(base_path, 'dev.jsonl'), os.path.join(base_path, 'test.jsonl')]
    outpath = os.path.join(base_path, 'songs_info.jsonl')
    extract_song_info(paths, outpath)

                