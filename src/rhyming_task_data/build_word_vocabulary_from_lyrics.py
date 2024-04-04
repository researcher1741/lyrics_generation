from argparse import ArgumentParser
from collections import Counter
import re
import jsonlines
import json
from tqdm import tqdm

def build_vocabulary(songs_path, out_path):
    vocabulary = Counter()
    pattern = re.compile(r'[;:\'"\]\}\{\[/\?\.>,<`~1!2@3#4$5%67890-=+_—\*\)\(&^%\$]+')

    with jsonlines.open(songs_path) as lines:
        for song in tqdm(lines, desc='songs'):
            lyrics = song['lyrics']
            sentences = lyrics.split('\n')
            for s in sentences:
                if s.startswith('[') and s.endswith(']'):
                    continue
                words = s.split(' ')
                last_idx = len(words) -1
                while re.matches(pattern, words[last_idx]):
                    last_idx -= 1
                                
                last_word = words[last_idx]
                vocabulary[last_word.lower()] += 1
    print('Dumping vocabulary')
    with open(out_path, 'w') as writer:
        json.dump(vocabulary, writer)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--song_path', required=True)
    parser.add_argument('--out_path', required=True)
    args = parser.parse_args()
    build_vocabulary(args.song_path, args.out_path)
    