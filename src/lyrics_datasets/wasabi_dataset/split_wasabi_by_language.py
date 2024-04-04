from argparse import ArgumentParser
import jsonlines
import os
from collections import Counter, defaultdict
from tqdm import tqdm
def split_by_language(path, out_dir):
    counter = Counter()
    songs = list()
    with jsonlines.open(path) as lines:
        for song in tqdm(lines, desc='reading'):
            songs.append(song)
            counter[song['lang']] += 1
    langs = set()
    print('counting songs by lang')
    for lang, count in counter.most_common(len(counter)):
        if len(lang) == "":
            continue
        if count >= 1000:
            langs.add(lang)
        else:
            break
    lang2writer = dict()
    print('writing songs by language')
    for song in tqdm(songs, desc='writing'):
        lang =song['lang']
        if lang not in langs:
            continue
        if lang not in lang2writer:
            lang2writer[lang] = jsonlines.open(os.path.join(out_dir, lang + '.jsonl'), 'w')
        lang2writer[lang].write(song)
    for writer in lang2writer.values():
        writer.close()
    print('all done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', default = '/lyrics_generation/data/wasabi/songs.jsonl')
    parser.add_argument('--out_dir', default='/lyrics_generation/data/wasabi/songs_by_language')
    args = parser.parse_args()
    split_by_language(args.path, args.out_dir)