for LANG in croatian danish dutch english finnish french german italian norwegian polish portuguese slovak spanish swedish turkish
do
    echo $LANG
    PYTHONPATH=src python src/lyrics_datasets/multilingual_processing/blockify_dataset.py \
        --path data/wasabi/songs_by_language/${LANG}.jsonl \
        --outpath data/wasabi/songs_by_language/section_dataset/${LANG}_section_dataset.jsonl \
        --language ${LANG}
done