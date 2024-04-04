import omegaconf
import hydra

import pytorch_lightning as pl
import os
import logging
import sys

from transformers import AutoTokenizer
from src.lyrics_generation.modules.pretraining_module import PretrainingModule
from src.lyrics_generation.data_modules.pretraining_data_module import PretrainingDataModule
from src.lyrics_generation_utils.utils import get_info_logger, set_pytorch_lightning_logger_level
os.environ['TOKENIZERS_PARALLELISM']='true'


def test(conf: omegaconf.DictConfig) -> None:
    generation_logger = get_info_logger('Generation Logger')
    if 'checkpoint_path' not in conf:
        generation_logger.error('out_dir or checkpoint_path are not in conf, please pass it when running the script as follows:\n\
        python generate.py +checkpoint_path=/path/to/checkpoint')
        sys.exit(1)
    pl.seed_everything(conf.train.seed)
    set_pytorch_lightning_logger_level(logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained(conf.model.tokenizer_path)
    pl_data_module = PretrainingDataModule(tokenizer, conf)
    current_conf_path='/tmp/hparams.yaml'
    with open(current_conf_path, 'w') as writer:
        writer.write(omegaconf.OmegaConf.to_yaml(conf))
    pl_module = PretrainingModule.load_from_checkpoint(conf.checkpoint_path, hparams_file=current_conf_path, strict=True)
    trainer: pl.Trainer = hydra.utils.instantiate(conf.train.pl_trainer)
    pl_data_module.setup(split_to_setup='test')
    trainer.test(pl_module, dataloaders=pl_data_module.test_dataloader())
    

@hydra.main(config_path="../../conf", config_name="root_pretrain_gpt2_zh2")
def main(conf: omegaconf.DictConfig):
    test(conf)


if __name__ == "__main__":
    main()
