import os
import torch
from src.lyrics_generation.data_modules.simple_data_module import SimpleDataModule
from src.lyrics_generation.modules.encoder_decoder_module import EncoderDecoderModule
from src.lyrics_generation.data_modules.multidataset_data_module import MultiDatasetDataModule

torch.multiprocessing.set_sharing_strategy('file_system')
# activate to simulate ROMA env
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
os.environ['MASTER_PORT'] = '9999'
import omegaconf
from omegaconf import OmegaConf
import hydra
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

import logging
from src.lyrics_generation_utils.utils import get_info_logger, get_lyrics_tokenizer
from src.lyrics_generation_utils.constants import LYRICS_SPECIAL_TOKENS
from omegaconf import open_dict

logger__ = logging.getLogger('transformers.utils.hub')
logger__.setLevel(logging.INFO)


def train(conf: omegaconf.DictConfig) -> None:
    console_logger = get_info_logger(__name__)

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
    # reproducibility
    pl.seed_everything(conf.train.seed)
    with open_dict(conf):
        conf.special_tokens = LYRICS_SPECIAL_TOKENS
        conf_str = OmegaConf.to_yaml(conf)
    aux = logging.getLogger("pytorch_lightning")
    aux.setLevel(logging.INFO)

    tokenizer = get_lyrics_tokenizer(conf)

    # if 'training_checkpoint_to_load' in conf.model and os.path.exists(conf.model.training_checkpoint_to_load):
    #     current_conf_path='/tmp/hparams.yaml'
    #     with open(current_conf_path, 'w') as writer:
    #         writer.write(OmegaConf.to_yaml(conf))
    #     pl_module = EncoderDecoderModule.load_from_checkpoint(conf.model.training_checkpoint_to_load, hparams_file=current_conf_path)
    #     console_logger.info(f'Loaded checkpoint {conf.model.training_checkpoint_to_load}')

    # else:
    pl_module = EncoderDecoderModule(conf)
    aux = logging.getLogger('datasets.arrow_dataset')
    aux.setLevel(logging.ERROR)
    # pl_data_module = SimpleDataModule(tokenizer, conf)
    pl_data_module = MultiDatasetDataModule(tokenizer, conf,
                                            decoder_start_token_id=pl_module.model.config.decoder_start_token_id)
    callbacks_store = []
    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(conf.train.model_checkpoint_callback)
        callbacks_store.append(model_checkpoint_callback)

    experiment_name = conf.train.model_name + '@' + conf.data.dataset_name
    log_path = '/'.join(os.getcwd().split('/')[:-3])
    notes = ''
    if 'description' in conf.train:
        notes = conf.train.description

    if 'WANDB_MODE' in os.environ and os.environ['WANDB_MODE'] == 'offline':
        # logger = True
        logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd())
    else:
        logger = pl_loggers.WandbLogger(name=experiment_name,
                                        save_dir=log_path, project='lyrics_generation',
                                        mode='dryrun' if dryrun else None, notes=notes,
                                        id=resume_id, resume=resume)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_store.append(lr_monitor)
    # trainer
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer, callbacks=callbacks_store, logger=logger,
                                               log_every_n_steps=500)
    tokenizer.save_pretrained(save_directory='tokenizer/')
    pl_data_module.setup()
    os.system('clear')
    print(conf_str)
    print('training_set:', len(pl_data_module.training_datasets['lyrics_generation']))
    print('dev_set:', len(pl_data_module.dev_datasets['lyrics_generation']))
    print('test_set:', len(pl_data_module.test_datasets['lyrics_generation']))
    if 'training_checkpoint_to_load' in conf.model:
        trainer.fit(pl_module, datamodule=pl_data_module, ckpt_path=conf.model.training_checkpoint_to_load)
    else:
        trainer.fit(pl_module, datamodule=pl_data_module)
