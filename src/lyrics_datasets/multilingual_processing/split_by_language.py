from argparse import ArgumentParser
import jsonlines
import os
from tqdm import tqdm


def split_by_language(path, out_dir, name_prefix):
    lang2writer = dict()
    with jsonlines.open(path) as lines:
        for line in tqdm(lines, 'splitting'):
            language = line['lang']
            writer = lang2writer.get(language)
            if writer is None:
                writer = jsonlines.open(os.path.join(out_dir, name_prefix + '.' + language + '.jsonl'), 'w')
                lang2writer[language] = writer
            writer.write(line)
    for writer in lang2writer.values():
        writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--name_prefix', required=True)
    args = parser.parse_args()
    split_by_language(args.path, args.out_dir, args.name_prefix)
