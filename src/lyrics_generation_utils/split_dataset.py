from argparse import ArgumentParser
import os
from random import shuffle
import jsonlines
from tqdm import tqdm
from src.lyrics_generation_utils.utils import get_info_logger
import random
random.seed(42)
def split_data(dataset_path, output_folder, train_split, dev_split, test_split):
    try:
        train_split = int(train_split)
        dev_split = int(dev_split)
        test_split = int(test_split)
    except:
        train_split = float(train_split)
        dev_split = float(dev_split)
        test_split = float(test_split)
    assert train_split is None or isinstance(train_split, int) or train_split + dev_split + test_split == 1.0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    logger = get_info_logger('Data Splitter')
    logger.info('reading songs')
    with jsonlines.open(dataset_path) as lines:
        all_songs = list(tqdm(iter(lines)))
    logger.info(f'loaded {len(all_songs)} songs')
    if train_split is None:
        assert type(dev_split) == type(test_split)
        if isinstance(dev_split, float):
            train_split = 1.0 - dev_split - test_split
        elif isinstance(dev_split, int):
            train_split = len(all_songs) - dev_split - test_split
    logger.info('shuffling')
    shuffle(all_songs)
    assert type(train_split) == type(dev_split) == type(test_split)
    if isinstance(train_split, float) and isinstance(dev_split, float):
        train_size = int(len(all_songs) * train_split)
        dev_size = int(len(all_songs) * dev_split)
        test_size = int(len(all_songs) * (1.0 - train_split - dev_split))
    else:
        assert isinstance(train_split, int) and isinstance(dev_split, int) and isinstance(test_split, int)
        train_size = train_split
        dev_size = dev_split 
        test_size = test_split
    assert train_size + dev_size + test_size <= len(all_songs)
    logger.info(f'All songs: {len(all_songs)}')
    logger.info(f'Training size: {train_size}')
    logger.info(f'Dev size: {dev_size}')
    logger.info(f'Test size: {test_size}')
    training_songs = all_songs[:train_size]
    dev_songs = all_songs[train_size: train_size + dev_size]
    test_songs = all_songs[train_size + dev_size: train_size + dev_size + test_size]
    for split, data in zip(['train', 'dev', 'test'], [training_songs, dev_songs, test_songs]):
        logger.info(f"Flushing {split}")
        with jsonlines.open(os.path.join(output_folder, split + '.jsonl'), 'w') as writer:
            for s in tqdm(data, desc=split):
                writer.write(s)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--train_split', default = None, help='If float, the parameter is used to cut a portion of the dataset, if an int, the parameter is used as exact number of examples to be sampled from the dataset.')
    parser.add_argument('--dev_split', default = None, help='If float, the parameter is used to cut a portion of the dataset, if an int, the parameter is used as exact number of examples to be sampled from the dataset.')
    parser.add_argument('--test_split', default = None, help='If float, the parameter is used to cut a portion of the dataset, if an int, the parameter is used as exact number of examples to be sampled from the dataset.')
    
    args = parser.parse_args()
    split_data(**vars(args))
