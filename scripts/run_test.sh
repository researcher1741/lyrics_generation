if [ "$#" -ne 2 ]; then
    echo 'Usage:'
    echo './run_test.sh /path/to/checkpoint /path/to/tokenizer'
    exit 1
fi

CKPT_PATH=`realpath $1`
TOKENIZER_PATH=`realpath $2`
echo "Loading checkpoint from: $CKPT_PATH"
echo "Loading tokenizer from: $TOKENIZER_PATH"


PYTHONPATH=src python src/lyrics_generation/evaluation/test.py \
    +checkpoint_path=\"$CKPT_PATH\" \
    model.from_checkpoint=True \
    ++model.tokenizer_path=\"$TOKENIZER_PATH\"
