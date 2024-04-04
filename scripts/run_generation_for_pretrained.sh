#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo 'Usage:'
    echo './run_generation.sh /path/to/checkpoint /path/to/tokenizer/'
    exit 1
fi
if [[ "$1" = /* ]]; then
    CKPT_PATH=$1
else
    CKPT_PATH=`realpath $1`
fi

if [[ "$2" = /* ]]; then
    TOKENIZER_PATH=$1
else
    TOKENIZER_PATH=`realpath $2`
fi



OUT_DIR=`echo $CKPT_PATH | sed "s/.ckpt//g"`

echo "Loading checkpoint from: $CKPT_PATH"
echo "Saving to dir: $OUT_DIR"

PYTHONPATH=src python src/lyrics_generation/evaluation/generate_from_pretrained.py +checkpoint_path=\"$CKPT_PATH\"\
    +out_dir=\"$OUT_DIR\"\
    model.from_checkpoint=True \
    model.tokenizer_path=\"$TOKENIZER_PATH\" \
