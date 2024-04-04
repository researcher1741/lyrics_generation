# Lyrics Generation

## Install
```bash
cd lyrics_generation
./setup.sh
```
This will ask you to insert the cuda version needed (11.1 or lower should work) and then proceed to install torch
and all the required dependences (in requirements.txt)

### Other Requirements
The project requires espeak-ng, a system library that can be installed as follows:
```bash
sudo apt install espeak-ng
```

However, for version of ubuntu <= 20.04, the version available in the repository is 1.49 which has a weird behaviour when it comes to Mandarine Chinese. To overcome this issue, one should install espeak-ng-1.50 from sources that can be found at this [link](https://github.com/espeak-ng/espeak-ng/releases/download/1.50/espeak-ng-1.50.tgz).
Follow installation instructions from this [page](https://github.com/espeak-ng/espeak-ng/blob/master/docs/building.md#linux-mac-bsd)

Once installation is completed, please add this line to your ~/.bashrc file:

```bash
export LD_LIBRARY_PATH=ESPEAK_LIB_PATH
```

If you ran installed from sources without chaning the target path, i.e., you ran `./configure; make; make install` your `ESPEAK_LIB_PATH` will be `/usr/local/lib`.

## Project Structure
The project is structured in he following folders:
- **conf** : contains the configuration files for the models, datasets, generation and training procedures.  
- **data** : contains the dataset used for training and testing 
- **src** : contains the source code of the project
- **scripts**: contains utility scripts to run training, generation and test.
- **notebooks**: contains notebooks used to browse datasets and compute statistics
- **experiments**: contains the checkpoints and the generations performed by varous models 
- **tokenizers**: contains config files for custom tokenizers

The whole project revolves around [pytorch-lightning](https://www.pytorchlightning.ai/) library and thus follows the suggested structure, i.e., models and datasets are represented by different classes and combined together within training files, e.g., `src/lyrics_generation/train_encoder_decoder.py`. The setup of models and dataset is done through configuration files, contained in the `conf/`, that are loaded through [hydra](https://hydra.cc/docs/intro/) library, which take care of combining different files.

## Configuration Files
The configuration folder is organised as follows:
```bash
conf/ # contains all root configuration file, one for each type of training, defining the data, the model and the training config files to use
├── data/ # contains a config file for each dataset with paths and other info about it
├── train/ # contains hyperparameters for the training
├── model/ # contains a config for each model with its parameters
└── generation/ # contains a config with generation parameters
```
Given the path to a root file (any in `conf/`) hydra reads all the other configuration files pointed out therein and return an `OmegaConf` object which allow accessing the various parameters through *dot notation*, i.e., `conf.data.training_path` allow to access the `training_path` variable within the `data` configuration file.

## Datasets
The dataset classes (I admit, are a bit convoluted due to their evolution during the research and could be semplified lot), allow to load different kind of datasets.
### Dataset Classes Hyeararchy:
- **LyricsDataset** (`src/lyrics_datasets/lyrics_dataset.py`): defines the logic to load, tokenize, filter and yield examples. In this dataset, each example is a song with its full lyrics and metadata (artist, title, etc.). 
- **LyricsBlockDataset** (`src/lyrics_datasets/lyrics_block_dataset.py`): this class extends `LyricsDataset`. The main difference is that each example is a block (i.e., paragraph, section) of a song, together with its metadata and, additionally, an optional text representing the preceeding block.
- **LyricsBlockDatasetLastWordFirst** (`src/lyrics_datasets/lyrics_block_dataset_last_word_first.py`): extends `LyricsBlockDataset` modifying only the way lyrics are tokenised, i.e., implementing the logic to organise each block verse with the last word as also the first word.


All in all, `LyricsDataset` is used when we want to train a full-song model, while `LyricsBlockDatasetLastWordFirst` when we want to train a paragraph-level model. 


### Dataset Class - Version mapping:
- LyricsDataset -> 0.1
- LyricsBlockDataset -> 0.2
- LyricsBlockDatasetLastWordFirst -> 0.2.1
- PretrainingDatasetEncoderDecoderLastWordFirst -> 0.2.2
- PretrainingDatasetEncoderDecoder -> 0.2.3

*In practice, `PretrainingDatasetEncoderDecoderLastWordFirst` extends `LyricsBlockDatasetLastWordFirst` slightly modifying the prompting. Same for `PretrainingDatasetEncoderDecoder` and `LyricsBlockDataset`. There is no real reason for this decoupling and those pair of classes can be merged together.*

## Training a model
To train a model one can use the scripts in `src/lyrics_generation/trainers/`. For example, to train a simple `T5` model with the `LyricsBlockDatasetLastWordFirst` dataset (which is the default) one can type in the following line:
```bash
PYTHONPATH=src python src/lyrics_genration/trainers/train_t5.py
```
This command will run the training of a T5 model as defined in `conf/root_t5.yaml`

The main training script for T5 is `src/lyrics_generation/trainers/train_encoder_decoder.py` and all the others call this one with a different configuration file. A similar behaviour happen for a decoder-only model, e.g., GPT2.

## Model Checkpoints
During training a model is evaluated every `x` steps as defined within the `train.yaml` config used. The checkpoints are saved in `./experiments/${data.dataset_name}-v${data.version}/${train.model_name}/` as defined in the chosen `root` config file. The `dataset_name`, `version` and `model_name` are variable defined in their corresponding config files and help to recognise which data and model name has been trained. For example, `experiments/genius_section-v0.2.1/t5v1.1-large` denotes the dataset used `genius_section-v0.2.1` and a `t5v1.1-large` model. Beside the checkpoint, in that directory there is also the `tokenizer` dump so that, at evaluation time, we can simply load that tokenizer for those checkpoints. This is needed as at training time some symbols are added to the tokenizer. 

## Generating Lyrics
To generate lyrics, one can use the script `scripts/run_generation.sh`. The script requires the path to the checkpoint and the tokenizer in order to run and will create a folder in the same folder of the passed checkpoint with its same name. Therein, it will create two files `generations.txt.` and `generations.txt.jsonl`. The first one can be used to human inspection, the second one is used to evaluate the rhyming capabilities of the model.

**N.B. when running the script, be sure that the right configuration file is set up in the `src/lyrics_generation/evaluation/generate.py`. The same config used during training has to be set, otherwise, there might be issues when loading the weights.**

## Computing Model Perplexity
To compute the perplexity of a model, similarly to when we run the generation, we can use the script `scripts/run_test.py` giving as input, again, the checkpoint and the tokenizer paths. Similarly to before, **be sure to set the proper configuration file in `src/lyrics_generation/evaluation/test.py`.

## Rhyme Evaluation
Once we let the model perform some generation on the test set inputs, we can evaluate its rhyming performance. To do so, we run the following command:
```bash
PYTHONPATH=src python src/lyrics_generation/evaluation/rhyming_evaluation2.py \
--path experiments/genius_section-v0.2.1/t5v1.1-large/checkpoint/epoch=11-step=399087.force_schema_True.ret_seq_20.do_sample_True/generations.txt.jsonl --language english
```

**N.B. we need to pass the language parameter, if we are using the multilingual model and the generations are in different languages, we can pass in `multi`**
