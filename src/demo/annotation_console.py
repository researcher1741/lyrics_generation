from argparse import ArgumentParser
import jsonlines
import os


def annotate(path, outpath):
    id_already_processed = set()

    if os.path.exists(outpath):
        with jsonlines.open(outpath) as lines:
            for line in lines:
                id_already_processed.add(line['_id'])
    lines_to_process = []
    with jsonlines.open(path) as lines:
        for line in lines:
            _id = line['_id']
            if _id in id_already_processed:
                continue
            lines_to_process.append(line)
    missing = len(lines_to_process)
    with jsonlines.open(outpath, 'a', flush=True) as writer:
        for i, line in enumerate(lines_to_process):
            print(f'{missing - i} annotations to go!')
            lyrics = line['lyrics']
            polished_lyrics = []
            for verse in lyrics.split('<sentence_end>'):
                polished_lyrics.append(verse.split('<sep>')[1])
            polished_lyrics = '\n'.join(polished_lyrics)
            print(polished_lyrics)
            print()
            while True:
                fluency = input('fluency (1 - 4)')
                try:
                    fluency = float(fluency)
                    break
                except:
                    print(fluency, 'value not valid, input a real number please!')
            while True:
                is_human = input('is human generated (y/n)').lower()
                if is_human not in {'y', 'n'}:
                    print(is_human, 'value is not valid, please type "y" for yes or "n" for no.')
                else:
                    is_human = is_human == 'y'
                    break
            line['fluency'] = fluency
            line['is_human'] = is_human
            writer.write(line)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lyrics_path', required=True)
    args = parser.parse_args()
    outpath = args.lyrics_path.replace('.jsonl', '.annotated.jsonl')
    annotate(args.lyrics_path, outpath)
