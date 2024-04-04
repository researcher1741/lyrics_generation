from argparse import ArgumentParser
import jsonlines
from src.lyrics_generation.evaluation.rhyming_eval_ori import evaluate

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--in_file', required=True)
    parser.add_argument('--out_file', required=True)
    args = parser.parse_args()

    languages = ['croatian', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'italian', 'norwegian', 'polish', 'portuguese', 'slovak', 'spanish', 'swedish', 'turkish']
    #languages = ['croatian', 'danish', 'dutch']
    f = jsonlines.open(args.folder + '/' + args.out_file, "w")
    for lang in languages:
        print('LANG:', lang)
        res = {'Language': lang.capitalize()}
        eval_d = evaluate(args.folder + '/' + args.in_file, language = lang)
        res.update(eval_d)
        print('=' * 30)
        f.write(res)
        
        
        
        
