import omegaconf
import hydra

import pytorch_lightning as pl
import os

import logging
import sys
from src.lyrics_generation_utils.utils import get_info_logger, set_pytorch_lightning_logger_level
from src.lyrics_generation_utils.init_utils import init_everything_for_evaluation

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

test_logger = get_info_logger('Lyrics Gen Test')


def test(conf: omegaconf.DictConfig) -> None:
    # if 'checkpoint_path' not in conf and 'test_out_file' not in conf:
    if 'checkpoint_path' not in conf:
        test_logger.error('checkpoint_path is not in conf, please pass it when running the script as follows:\n\
        python test.py +checkpoint_path=/path/to/checkpoint')
        sys.exit(1)

    # checkpoint_path = conf.checkpoint_path

    # reproducibility
    pl.seed_everything(conf.train.seed)
    set_pytorch_lightning_logger_level(logging.INFO)

    # data module declaration
    trainer, pl_module, pl_data_module, _ = init_everything_for_evaluation(conf)

    print(pl_module.device)
    # module test
    trainer.test(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../../../conf", config_name="root_multitask_t5")
def main(conf: omegaconf.DictConfig):
    test(conf)


if __name__ == "__main__":
    main()
