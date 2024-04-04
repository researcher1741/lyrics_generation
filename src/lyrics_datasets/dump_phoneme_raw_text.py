import jsonlines
from tqdm import tqdm


def dump_phoneme_raw_text(in_dataset_path, out_path):
    with jsonlines.open(in_dataset_path) as lines, open(out_path, 'w') as writer:
        for song in tqdm(lines):
            blocks = song['espeak_tokenized_lyrics']
            if blocks is None:
                continue
            phonemes = list()
            for block in blocks:
                for sentence in block:
                    phonemes.extend([w.replace('.', '') for w in sentence])
            if len(phonemes) > 0:
                writer.write(' '.join(phonemes) + '\n')


if __name__ == '__main__':
    dump_phoneme_raw_text('data/genius/phonemised_dataset/phonemised_dataset.jsonl',
                          'data/genius/phonemised_dataset/phonemes.txt')
