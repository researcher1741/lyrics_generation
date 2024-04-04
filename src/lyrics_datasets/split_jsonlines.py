from argparse import ArgumentParser
from random import shuffle
import jsonlines
import os
from tqdm import tqdm


def dump(data, outpath):
    with jsonlines.open(outpath, 'w') as writer:
        for elem in data:
            writer.write(elem)


def split(path, out_dir, do_shuffle=True, train_split=0.9):
    assert 1.0 > train_split > 0
    validation_split = (1 - train_split) / 2
    data = []
    with jsonlines.open(path) as lines:
        for line in tqdm(lines, desc='reading'):
            data.append(line)
    if do_shuffle:
        shuffle(data)
    train_size = int(len(data) * train_split)
    dev_size = int(len(data) * validation_split)
    training = data[:train_size]
    dev = data[train_size:train_size + dev_size]
    test = data[train_size + dev_size:]
    print('dumping training')
    dump(training, os.path.join(out_dir, 'train.jsonl'))
    print('dumping dev')
    dump(dev, os.path.join(out_dir, 'dev.jsonl'))
    print('dumping test')
    dump(test, os.path.join(out_dir, 'test.jsonl'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--out_dir')
    args = parser.parse_args()
    split(args.path, args.out_dir)
