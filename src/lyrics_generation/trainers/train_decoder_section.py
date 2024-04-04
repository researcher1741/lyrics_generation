import os
from src.lyrics_generation.data_modules.multidataset_data_module import MultiDatasetDataModule
from src.lyrics_generation.modules.decoder_module import DecoderModule

# activate to simulate ROMA env
import os
os.environ['TOKENIZERS_PARALLELISM']='true'
os.environ['REQUESTS_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'
os.environ['MASTER_PORT'] = '9999'
os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['TRANSFORMERS_OFFLINE']='1'
os.system('pwd')
import omegaconf
from omegaconf import OmegaConf
import hydra

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import  ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import logging
from src.lyrics_generation_utils.utils import get_info_logger, get_lyrics_tokenizer
# from src.lyrics_generation_utils.constants import LYRICS_SPECIAL_TOKENS
from src.lyrics_generation_utils.constants import LYRICS_SPECIAL_TOKENS
from omegaconf import open_dict
logger__ = logging.getLogger('transformers.utils.hub')
logger__.setLevel(logging.INFO)

def train(conf: omegaconf.DictConfig) -> None:
    if 'dryrun' in conf:
        dryrun = conf.dryrun
    else:dryrun = False
    # reproducibility
    pl.seed_everything(conf.train.seed)
    with open_dict(conf):
        conf.special_tokens = LYRICS_SPECIAL_TOKENS
    aux = logging.getLogger("pytorch_lightning")
    aux.setLevel(logging.INFO)
    console_logger = get_info_logger(__name__)
    print(OmegaConf.to_yaml(conf))
    

    tokenizer = get_lyrics_tokenizer(conf)

    if 'training_checkpoint_to_load' in conf.model and os.path.exists(conf.model.training_checkpoint_to_load):
        current_conf_path='/tmp/hparams.yaml'
        with open(current_conf_path, 'w') as writer:
            writer.write(OmegaConf.to_yaml(conf))
        pl_module = DecoderModule.load_from_checkpoint(conf.model.training_checkpoint_to_load, hparams_file=current_conf_path)
        console_logger.info(f'Loaded checkpoint {conf.model.training_checkpoint_to_load}')
        
    else:
        pl_module = DecoderModule(conf)
    assert conf.data.version == '0.2.1'
    pl_data_module = MultiDatasetDataModule(tokenizer, decoder_start_token_id = None, conf= conf)
    # pl_data_module.prepare_data()
    callbacks_store = []
    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(conf.train.model_checkpoint_callback)
        callbacks_store.append(model_checkpoint_callback)

    experiment_name = conf.train.model_name + '@' + conf.data.dataset_name
    log_path = '/'.join(os.getcwd().split('/')[:-2])
    notes = ''
    if 'description' in conf.train:
        notes = conf.train.description
    logger = pl_loggers.WandbLogger(name=experiment_name, 
            save_dir = log_path, project='lyrics_generation', 
            mode='dryrun' if dryrun else None, notes=notes)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer, callbacks=callbacks_store, logger=logger)
    tokenizer.save_pretrained(save_directory='tokenizer/')

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

@hydra.main(config_path="../../../conf", config_name="root_gpt2_section")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
