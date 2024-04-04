bash
conda init bash
rest
bash
conda activate /home/ma-user/anaconda3/envs/Pytorch-1.0.0
(echo "import moxing as mox"; echo "mox.file.copy_parallel('s3://lyrics_generation/src', 'src')") | python
mkdir -p data/genius/phonemised_dataset/
(echo "import moxing as mox"; echo "mox.file.copy_parallel('s3://lyrics_generation/data/genius/phonemised_dataset/section_dataset_rhyme_and_verse_length_filtered/', 'data/genius/phonemised_dataset/section_dataset_rhyme_and_verse_length_filtered/')") | python
(echo "import moxing as mox"; echo "mox.file.copy_parallel('s3://lyrics_generation/conf', 'conf')") | python
(echo "import moxing as mox"; echo "mox.file.copy_parallel('s3://lyrics_generation/data/requirements.txt', 'requirements.txt')") | python

pip install pip --upgrade
pip install torch --upgrade
pip install packaging==20.9
pip install transformers -U --ignore-installed
pip install pandas==0.24.2
pip install pytorch-lightning
pip install sentencepiece 
pip install -r requirements.txt --retries 10