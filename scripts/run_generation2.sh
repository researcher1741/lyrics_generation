#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo 'Usage:'
    echo './run_generation.sh /path/to/checkpoint /path/to/tokenizer/ [--force_rhyming_schema]'
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
FORCE_SCHEMA=False
if [ "$#" -eq 3 ]; then 
    if [ "$3" = "--force_schema" ]; then # check if --force_rhyming_schema
        FORCE_SCHEMA=True
    fi
fi



OUT_DIR=`echo $CKPT_PATH | sed "s/.ckpt/.force_schema_$FORCE_SCHEMA/g"`

echo "Loading checkpoint from: $CKPT_PATH"
echo "Saving to dir: $OUT_DIR"

PYTHONPATH=src python src/lyrics_generation/evaluation/generate2.py +checkpoint_path=\"$CKPT_PATH\"\
    +out_dir=\"$OUT_DIR\"\
    model.from_checkpoint=True \
    ++model.tokenizer_path=\"$TOKENIZER_PATH\" \
    model.force_schema=$FORCE_SCHEMA
