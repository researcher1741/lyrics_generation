from argparse import ArgumentParser
import jsonlines
from lyricsgenius import Genius
from multiprocessing import Pool
from tqdm import tqdm
import sys
import os

TOKEN1 = 'lqAXJ-N6bBbm8POn5KW3106DjRBHKMHlSjgRu5B7UCpvcqNziGmcelnRmOymQiOt'
TOKEN2 = 'hwdtyD4VkxKI4cQvmD9X7hYue5QGyM6Z9rV21rkiKqIMwOLJAMAioMcmjC04VlYc'
TOKEN3 = 'pOD4ch1B2fLpFKVfzMUS-GYG1HKWOfVnQKETqnimR1lRV_3MGF20ewxXrlHTHDK5'

TOKEN = None

def download_song(item):
    artist_name, song_limit = item
    genius = Genius(TOKEN, verbose=False, timeout=10)
    try:
        artist = genius.search_artist(artist_name, max_songs=song_limit)
    except Exception as e:
        print(f'[ERROR] Cannot download songs for {artist_name}.')
        return []
    songs = list()
    if artist is None:
        return songs
    for song in artist.songs:
        if song is None:
            continue
        lyrics = song.lyrics

        title = song.title
        songs.append({'title': title, 'lyrics': lyrics, 'artist': song.artist})
    return songs
        
def download_songs(artist_list, outpath, lang, songs_limit):
    mod = 'w'
    if os.path.exists(outpath):
        mod = 'a'
    if mod == 'w':
        print(f'Writing songs to {outpath}')
    else:
        print(f'Appending songs to {outpath}')
    i = input(f"You sure you want to open {outpath} with mod {mod}?(y/n)")
    if i == 'n':
        sys.exit(1)
    pool = Pool(1)
    aux = zip(artist_list, [songs_limit] * len(artist_list))
    songs = pool.imap_unordered(download_song, aux)
    

    with jsonlines.open(outpath, mod) as writer:
        for batch_songs in tqdm(songs):
            if len(batch_songs) > 0:
                writer.write(batch_songs)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--artist_list_path', default='data/top_1000_spotify_artists.txt')
    parser.add_argument('--outpath', required=True)
    parser.add_argument('--data_split', nargs='+', default = (0, 333), type=int)
    parser.add_argument('--client_id', type=int, choices=[1,2,3])
    parser.add_argument('--lang', required=True)
    parser.add_argument('--artist_songs_limit', default=None, type=int)
    
    
    
    args = parser.parse_args()
    client_id = args.client_id
    if client_id == 1:
        TOKEN = TOKEN1
    elif client_id == 2:
        TOKEN = TOKEN2
    elif client_id == 3:
        TOKEN = TOKEN3
    else:
        print(f'CLIENT_ID "{client_id}" not recognised. Use one among [1, 2, 3]')
        sys.exit(1)
    print(f'Downloading artists from {args.data_split[0]} to {args.data_split[1]}')
    print(f'Client Id: {client_id}')
    data_split = args.data_split
    artist_list_path = args.artist_list_path
    processed_artists = set()
    if os.path.exists(args.outpath):
        with jsonlines.open(args.outpath) as lines:
            for lst in lines:
                for s in lst:
                    processed_artists.add(s['artist'])
    print(f'Found {len(processed_artists)} already-processed artists, gonna skip them')
    artists = list()
    with open(artist_list_path) as lines:
        next(lines)
        for i, line in enumerate(lines):
            if i >= data_split[1]:
                break
            if i >= data_split[0]:
                fields = line.strip().split('\t')
                idx = 0
                if len(fields) > 1:
                    idx = 1
                if fields[idx] in processed_artists:
                    continue
                artists.append(fields[idx])

    print(f'Loaded {len(artists)} to download...')
    download_songs(artists, args.outpath, args.lang, args.artist_songs_limit)
