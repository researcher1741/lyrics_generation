import argparse
import os
from argparse import ArgumentParser

def report_dir_info():
    print('Current directory')
    os.system('pwd')
    print()
    print("Content:")
    os.system('ls')
    print()
    print('Content data url')
    os.system(f'ls {args.data_url}')
    print()

def install_dependencies(data_dir):
    print('Testing pip...')
    os.system('pip install --upgrade pip')
    print()
    print('Installing dependencies...')
    os.system(f'pip install -r {data_dir}/requirements.txt --retries 10')
    
def link_dirs(args):
    print('linking conf dir')
    os.system(f'ln -s {args.config_url} ./conf')
    print('linking data dir')
    os.system(f'ln -s {args.data_url} ./data')
    # print('linking experiment dir')
    # os.system(f'ln -s {args.train_url} ./experiments')
    print('linking models dir')
    os.system(f'ln -s {args.models_url} ./models')
    print('ls')
    os.system('ls')
    os.system('ls -R data/')

if __name__ == '__main__':
    argparse = ArgumentParser()
    argparse.add_argument('--data_url', default=None)
    argparse.add_argument('--config_url', default=None)
    argparse.add_argument('--models_url', default=None)
    argparse.add_argument('--init_method', default=None)
    # argparse.add_argument('--model_name', default=None)

    args = argparse.parse_args()
    print('FORZA ROMA')
    os.system('nvidia-smi')
    print('data url', args.data_url)
    
    report_dir_info()
    
    os.system('mv ~/.pip/pip.conf ~/.pip/pip.conf.backup')
    os.system(f'cp {args.data_url}/pip.conf ~/.pip/')
    os.system(f'cat {os.path.join(args.data_url, "pip.conf")}')
    src_path = os.path.join(os.getcwd(), 'src/')
    install_dependencies(args.data_url)
    link_dirs(args)
    os.system('file src/lyrics_generation/trainers/train_roma.py')
    # os.system('PYTHONPATH=src python src/lyrics_generation/trainers/train_roma.py')
    os.system('bash src/roma_scripts/train.sh')
    # if args.model_name == 'mt5':
    #     os.system(f'')
    # if args.model_name == 't5':
    #     os.system(f'bash src/roma_scripts/train-t5.sh')
    
    # if args.model_name == 'custom-t5':
    #     os.system(f'bash src/roma_scripts/train-custom-t5.sh')

    # if args.model_name == 'multitask-t5':
    #     os.system(f'bash src/roma_scripts/train-multitask-t5.sh')

    # if args.model_name == 'custom-multitask-t5':
    #     os.system(f'bash src/roma_scripts/train-custom-multitask-t5.sh')
