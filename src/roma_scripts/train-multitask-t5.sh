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

PYTHONPATH=src python src/lyrics_generation/train_multitask_t5.py \
    data.train_path="/lyrics_generation/DATA/Lyrics_English_Section_Dataset_Rhyme/train.jsonl" \
    data.validation_path="/lyrics_generation/DATA/Lyrics_English_Section_Dataset_Rhyme/dev.jsonl"\
    data.test_path="/lyrics_generation/DATA/Lyrics_English_Section_Dataset_Rhyme/test.jsonl" \
    data.genre_mapping_path="/home/ma-user/modelarts/user-job-dir/data/genres_mapping_100.txt"\
    data.word_frequency_data_path="/home/ma-user/modelarts/user-job-dir/data/enwiki-20210820-words-frequency.filtered.pkl"\
    train.pl_trainer.gpus="1"\
    train.pl_trainer.enable_progress_bar="False"
    