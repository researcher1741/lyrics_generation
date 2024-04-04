from argparse import ArgumentParser
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('checkpoint_path')
    parser.add_argument('tokenizer_path')
    parser.add_argument('--out_path', default=None)
    parser.add_argument('--config_overrides', nargs='+', default=[])
    args = parser.parse_args()

    CKPT_PATH = os.path.realpath(args.checkpoint_path)
    TOKENIZER_PATH = os.path.realpath(args.tokenizer_path)
    if args.out_path is None:
        OUT_DIR = CKPT_PATH.replace('.ckpt', '')
    else:
        OUT_DIR = args.out_path

    CONFIG_OVERRIEDS = ' '.join(args.config_overrides)
    CKPT_PATH = CKPT_PATH.replace("=", "\\=")
    TOKENIZER_PATH = TOKENIZER_PATH.replace('=', '\\=')
    OUT_DIR = OUT_DIR.replace('=', '\\=')
    
    print('Checkpoint Path:', CKPT_PATH)
    print('Tokenizer Path:', TOKENIZER_PATH)
    print('Out Path:', OUT_DIR)
    print('Config Overrides:', CONFIG_OVERRIEDS)

    COMMAND = f'''PYTHONPATH=src python src/lyrics_generation/generate.py +checkpoint_path="{CKPT_PATH}"\
        +out_dir="{OUT_DIR}"\
        model.from_checkpoint=True\
        +model.tokenizer_path="{TOKENIZER_PATH}"\
        {CONFIG_OVERRIEDS}
        '''
    print(COMMAND)
    os.system(COMMAND)