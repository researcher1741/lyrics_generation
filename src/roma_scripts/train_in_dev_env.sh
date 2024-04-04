export WANDB_API_KEY=82a3a92497ed6a3a52d55c347754666e2554f3cd
export WANDB_ENTITY=pasini
export WANDB_PROJECT=lyrics_generation
export HYDRA_FULL_ERROR=1


PYTHONPATH=src python src/lyrics_generation/train_encoder_decoder.py \
    data.train_path="/home/ma-user/work/data/genius/phonemised_dataset/section_dataset_rhyme_and_verse_length_filtered/train.jsonl" \
    data.validation_path="/home/ma-user/work/data/genius/phonemised_dataset/section_dataset_rhyme_and_verse_length_filtered/dev.jsonl"\
    data.test_path="/home/ma-user/work/data/genius/phonemised_dataset/section_dataset_rhyme_and_verse_length_filtered/test.jsonl" \
    data.genre_mapping_path="/home/ma-user/work/data/genres_mapping_100.txt"\
    data.word_frequency_data_path="/home/ma-user/work/data/enwiki-20210820-words-frequency.filtered.pkl"\
    train.pl_trainer.gpus="1"