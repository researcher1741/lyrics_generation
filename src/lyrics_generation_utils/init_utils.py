import os
import hydra
from pytorch_lightning import Trainer
from transformers import AutoConfig
from src.lyrics_generation.data_modules.decoder_data_module import DecoderDataModule
from src.lyrics_generation.data_modules.multidataset_data_module import MultiDatasetDataModule
from src.lyrics_generation.modules.encoder_decoder_module import EncoderDecoderModule
from src.lyrics_generation.modules.decoder_module import DecoderModule
from src.lyrics_generation.modules.multitask_lyrics_module import MultitaskLyricsModule

from src.lyrics_generation_utils.utils import DotDict, get_info_logger, get_lyrics_tokenizer


def init_everything_for_evaluation(conf, disable_logger=False):
    print('DISABLE LOGGER', disable_logger)
    if not disable_logger:
        print('get info logger')
        logger = get_info_logger(__name__)
    else:
        logger = DotDict({'info': print})
    from omegaconf import OmegaConf
    current_conf_path=f'/tmp/{os.getlogin()}_hparams.yaml'
    with open(current_conf_path, 'w') as writer:
        writer.write(OmegaConf.to_yaml(conf))
    logger.info(str(conf))
    tokenizer = get_lyrics_tokenizer(conf)
    logger.info(f"Tokenizer Loaded: {len(tokenizer)}'")
    hf_conf = AutoConfig.from_pretrained(conf.model.pretrained_model_name_or_path)
    model_cls = None
    if hf_conf.is_encoder_decoder:
        logger.info('Loding Encoder Decoder Model')
        if 'multitask' in conf.model.pretrained_model_name_or_path:
            model_cls = MultitaskLyricsModule
        else:
            model_cls = EncoderDecoderModule
    else:
        logger.info('Loading Decoder Model')
        model_cls = DecoderModule
    pl_module = model_cls.load_from_checkpoint(conf.checkpoint_path, hparams_file=current_conf_path, map_location='cpu', strict=True)
    data_cls = None
    if hf_conf.is_encoder_decoder:
        logger.info('Loading from MultiDatasetDataModule (LM Task + Other Tasks)')
        data_cls = MultiDatasetDataModule
    else:
        logger.info('Loading from DecoderDataModule (LM Task)')
        data_cls = DecoderDataModule
    pl_data_module = data_cls(tokenizer, conf, decoder_start_token_id = pl_module.model.config.decoder_start_token_id)

    pl_module = pl_module.eval()
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer)
    return trainer, pl_module, pl_data_module, tokenizer