import omegaconf
import hydra

import pytorch_lightning as pl
import os
import logging
import sys

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.lyrics_generation.modules.pretraining_module import PretrainingModule
from src.lyrics_generation.evaluation.generate import run_generation
from src.lyrics_generation.data_modules.pretraining_data_module import PretrainingDataModule
from src.lyrics_generation_utils.utils import get_info_logger, set_pytorch_lightning_logger_level
os.environ['TOKENIZERS_PARALLELISM']='true'


def generate(conf: omegaconf.DictConfig) -> None:
    generation_logger = get_info_logger('Generation Logger')
    if 'checkpoint_path' not in conf and 'out_dir' not in conf:
        generation_logger.error('out_dir or checkpoint_path are not in conf, please pass it when running the script as follows:\n\
        python generate.py +checkpoint_path=/path/to/checkpoint +out_dir=/path/to/outdir/ [model.force_schema=[True,False]')
        sys.exit(1)
    out_dir = conf.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # reproducibility
    pl.seed_everything(conf.train.seed)
    set_pytorch_lightning_logger_level(logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained(conf.model.tokenizer_path)
    pl_data_module = PretrainingDataModule(tokenizer, conf)

    current_conf_path='/tmp/hparams2.yaml'
    with open(current_conf_path, 'w') as writer:
        writer.write(OmegaConf.to_yaml(conf))
    language = conf.data.language
    pl_module = PretrainingModule.load_from_checkpoint(conf.checkpoint_path, hparams_file=current_conf_path, strict=True)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    gen_args = {'eos_token_id': eos_token_id, 
                'pad_token_id': tokenizer.pad_token_id}
    run_generation(conf, pl_module, pl_data_module, tokenizer, language, **gen_args)
    

@hydra.main(config_path="../../../conf", config_name="root_pretrain_gpt2_zh")
def main(conf: omegaconf.DictConfig):
    generate(conf)


if __name__ == "__main__":
    main()
