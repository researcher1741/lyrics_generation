
import os
from argparse import ArgumentParser

from transformers import AutoTokenizer, PreTrainedTokenizer
from lyrics_generation_utils.utils import LYRICS_SPECIAL_TOKENS


def train_tokenizer(model, path, outpath, vocab_size):
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(model)
    new_tokenizer = tokenizer.train_new_from_iterator(iter(open(path)), vocab_size, new_special_tokens=LYRICS_SPECIAL_TOKENS)
    new_tokenizer.save_pretrained(outpath)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', default='data/genius/phonemised_dataset/phonemes.txt')
    parser.add_argument('--vocab_size', default=10_000, type=int)
    parser.add_argument('--out_path', default='data/genius/phonemised_dataset/gpt2-medium_phoneme_tokenizer/')
    parser.add_argument('--model', default='gpt2-medium')
    parser.add_argument('--max_len', default=1024)
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    train_tokenizer(args.model, args.path, args.out_path, args.vocab_size)