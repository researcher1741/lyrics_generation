import omegaconf
import hydra

import pytorch_lightning as pl
import os

import logging
import sys
from src.lyrics_generation_utils.utils import get_info_logger, set_pytorch_lightning_logger_level
from src.lyrics_generation_utils.init_utils import init_everything_for_evaluation

from hydra import initialize, compose
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

CHECKPOINT_PATH='./LG/CHECKPOINTS/T5_large_EN/epoch=9-step=250786.ckpt'
TOKENIZER_PATH='./LG/CHECKPOINTS/T5_large_EN/tokenizer'
CONFIG_NAME='root_t5'

test_logger = get_info_logger('Lyrics Gen Test')

def test() -> None:
    with initialize(config_path="../../../conf", ):
        conf = compose(config_name=CONFIG_NAME, overrides=[f'+checkpoint_path="{CHECKPOINT_PATH}"', f'+model.tokenizer_path="{TOKENIZER_PATH}"'])
    
    # reproducibility
    pl.seed_everything(conf.train.seed)
    set_pytorch_lightning_logger_level(logging.INFO)
    
    trainer, pl_module, pl_data_module, _ = init_everything_for_evaluation(conf)
    
    print(pl_module.device)
    # module test
    trainer.test(pl_module, datamodule=pl_data_module)

#@hydra.main(config_path="../../../conf", config_name="root_t5.yaml")
def main():
    test()


if __name__ == "__main__":
    main()
