import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import os
from transformers import AutoTokenizer
import logging
from src.lyrics_generation_utils.modules.pretraining_module import PretrainingModule
from src.lyrics_generation_utils.data_modules.pretraining_data_module import PretrainingDataModule
from src.lyrics_generation_utils.utils import LYRICS_SPECIAL_TOKENS, get_info_logger
from omegaconf import open_dict

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'


def train(conf: omegaconf.DictConfig) -> None:
    console_logger = get_info_logger(__name__)

    # reproducibility
    pl.seed_everything(conf.train.seed)
    if 'dryrun' in conf:
        dryrun = conf.dryrun
    else:
        dryrun = False
    if 'resume_id' in conf:
        resume_id = conf.resume_id
        resume = 'must'
        console_logger.info(f"WanDB resuming from {resume_id}")
    else:
        resume_id = None
        resume = False
    with open_dict(conf):
        conf.special_tokens = LYRICS_SPECIAL_TOKENS

    logger = logging.getLogger("pytorch_lightning")
    logger.setLevel(logging.INFO)

    # data module declaration
    assert 'tokenizer_path' in conf.model or logger.warn(
        'Cannot proceed without having a specified tokenizer in model configuration. Please define tokenizer_path key in model config.')
    tokenizer = AutoTokenizer.from_pretrained(conf.model.tokenizer_path)
    pl_data_module = PretrainingDataModule(tokenizer, conf)

    # main module declaration
    pl_module = PretrainingModule(conf)
    if 'checkpoint_path_to_load' in conf.train and os.path.exists(
            conf.train.checkpoint_path_to_load) and conf.train.resume_training \
            and conf.model.tokenizer_path:
        logger.info(f'Resuming Training from {conf.train.checkpoint_path_to_load}')
        tokenizer = AutoTokenizer.from_pretrained(conf.model.tokenizer_path)
        pl_module.load_from_checkpoint(conf.train.checkpoint_path_to_load)

    # callbacks declaration
    callbacks_store = []

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(conf.train.model_checkpoint_callback)
        callbacks_store.append(model_checkpoint_callback)

    experiment_name = conf.train.model_name + '@' + conf.data.dataset_name
    log_path = '/'.join(os.getcwd().split('/')[:-2])
    notes = ''
    if 'description' in conf.train:
        notes = conf.train.description

    if 'WANDB_MODE' in os.environ and os.environ['WANDB_MODE'] == 'offline':
        logger = True
    else:
        logger = pl_loggers.WandbLogger(name=experiment_name,
                                        save_dir=log_path, project='lyrics_generation',
                                        mode='dryrun' if dryrun else None, notes=notes,
                                        id=resume_id, resume=resume)
    # trainer
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer, callbacks=callbacks_store, logger=logger)
    tokenizer.save_pretrained(save_directory='tokenizer/')
    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)


## TODO when loading the config for GPT2 reassign the vocab_size to match the one of the vocabulary.

@hydra.main(config_path="../../conf", config_name="root_pretrain_gpt2_zh")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
