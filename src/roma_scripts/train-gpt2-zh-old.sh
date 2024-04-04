#!/bin/bash
echo 'IN TRAIN.sh'
echo `pwd`
echo `ls`

export TOKENIZERS_PARALLELISM='true'
export TRANSFORMERS_OFFLINE='1'
export REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'
export HF_HOME="`pwd`/.cache/huggingface"
export WANDB_API_KEY='82a3a92497ed6a3a52d55c347754666e2554f3cd'
export WANDB_MODE='offline'

PYTHONPATH=src python src/lyrics_generation/train_custom_t5.py \
    data.train_path='/home/ma-user/work/rhyming_data/train.jsonl' \
    data.validation_path='/home/ma-user/work/rhyming_data/dev.jsonl'\
    data.test_path='/home/ma-user/work/rhyming_data/test.jsonl' \
    data.genre_mapping_path="/home/ma-user/work/data/genres_mapping_zh_old.txt"\
    data.year_mapping_path="/home/ma-user/work/data/year_mapping_zh.txt"\
    data.word_frequency_data_path="/home/ma-user/work/data/enwiki-20210820-words-frequency.filtered.pkl"\
    train.pl_trainer.gpus="1"\
    train.pl_trainer.enable_progress_bar="False"