from argparse import ArgumentParser
from tqdm import tqdm
import jsonlines

def is_number(ch):
    try:
        int(ch)
    except:
        return False
    return True

def merge_downloads(outpath, *files):
    songs = list()
    title_artist_set = set()
    for f in tqdm(files):
        with jsonlines.open(f) as lines:
            for artist_songs in lines:
                for s in artist_songs:
                    if s['title'] + '#' + s['artist'] in title_artist_set:
                        continue
                    lyrics = s['lyrics'].strip()
                    if lyrics.endswith('Embed'):
                        lyrics = lyrics[:-len('embed') - 1]
                        while is_number(lyrics[-1]):
                            lyrics = lyrics[:-1]
                        s['lyrics'] = lyrics
                    songs.append(s)
                    title_artist_set.add(s['title'] + '#' + s['artist'])
    print('dumping')
    with jsonlines.open(outpath, 'w') as writer:
        for s in tqdm(songs):
            writer.write(s)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--files', nargs='+', default=['data/genius_lyrics_0_333.jsonl', 'data/genius_lyrics_333_666.jsonl', 'data/genius_lyrics_666_1000.jsonl'])
    parser.add_argument('--outpath', type=str, default='data/genius_lyrics.jsonl')
    args = parser.parse_args()
    merge_downloads(args.outpath, *args.files)