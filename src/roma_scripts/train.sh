#!/bin/bash
echo 'IN TRAIN.sh'
echo `pwd`
echo `ls`

export TOKENIZERS_PARALLELISM='true'
export TRANSFORMERS_OFFLINE='1'
export REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'
# export HF_HOME="`pwd`/.cache/huggingface"
export WANDB_API_KEY='82a3a92497ed6a3a52d55c347754666e2554f3cd'
export WANDB_MODE='offline'

# python -c "print('Cane')"
export PYTHONPATH=src 
python src/lyrics_generation/trainers/train_roma.py