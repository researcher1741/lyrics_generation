from transformers import AutoConfig, AutoModelForCausalLM, MT5Config, MT5ForConditionalGeneration, T5Config, \
    T5ForConditionalGeneration

from src.lyrics_generation_utils.utils import get_info_logger
from src.lyrics_models.custom_t5 import CustomT5ForConditionalGeneration
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


def get_mt5_for_conditional_generation(model_name, from_pretrained, embedding_len, **kwargs):
    logger = get_info_logger(__name__)
    logger.info(f'loading model {model_name}')
    if from_pretrained:
        model = MT5ForConditionalGeneration.from_pretrained(model_name, **kwargs)
    else:
        conf = MT5Config.from_pretrained(model_name)
        model = MT5ForConditionalGeneration(conf)

    model_embeddings_size = model.lm_head.out_features
    if embedding_len != model_embeddings_size:
        logger.info(
            f'Resizing model embedding to match tokenizer size. Model: {model_embeddings_size}, Tokenizer: {embedding_len}')
        model.resize_token_embeddings(embedding_len)

    return model


def get_automodel_for_conditional_generation(model_name, embedding_len, from_pretrained, **kwargs):
    logger = get_info_logger(__name__)
    logger.info(f'loading model (auto) {model_name}')
    if from_pretrained:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        conf = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM(conf)

    model_embeddings_size = model.lm_head.out_features
    if embedding_len != model_embeddings_size:
        logger.info(
            f'Resizing model embedding to match tokenizer size. Model: {model_embeddings_size}, Tokenizer: {embedding_len}')
        model.resize_token_embeddings(embedding_len)

    return model


def get_t5_for_conditional_generation(model_name, embedding_len, from_pretrained, **kwargs):
    logger = get_info_logger(__name__)
    logger.info(f'loading model (t5) {model_name}')
    if from_pretrained:
        model = T5ForConditionalGeneration.from_pretrained(model_name, **kwargs)
    else:
        conf = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(conf)

    model_embeddings_size = model.lm_head.out_features
    if embedding_len != model_embeddings_size:
        logger.info(
            f'Resizing model embedding to match tokenizer size. Model: {model_embeddings_size}, Tokenizer: {embedding_len}')
        model.resize_token_embeddings(embedding_len)

    return model


def get_custom_t5_for_conditional_generation(model_name, embedding_len, from_pretrained, conf, **kwargs):
    logger = get_info_logger(__name__)
    logger.info(f'loading model (custom_t5) {model_name}')
    model_config = T5Config.from_pretrained(model_name)
    model_config.max_num_rhymes = conf.model.max_num_rhymes
    model_config.max_sentence_len = conf.model.max_sentence_len
    if from_pretrained:
        model = CustomT5ForConditionalGeneration.from_pretrained(model_name, config=model_config, **kwargs)
    else:
        model = CustomT5ForConditionalGeneration(model_config)

    model_embeddings_size = model.lm_head.out_features
    if embedding_len != model_embeddings_size:
        logger.info(
            f'Resizing model embedding to match tokenizer size. Model: {model_embeddings_size}, Tokenizer: {embedding_len}')
        model.resize_token_embeddings(embedding_len)

    return model


def get_lyrics_modelling_model(model_name, tokenizer, from_pretrained, conf=None, **kwargs):
    tokenizer_len = len(tokenizer)
    use_custom = False
    if conf is not None and hasattr(conf.model, 'use_custom'):
        use_custom = conf.model.use_custom
    if 'mt5' in model_name:
        if use_custom:
            raise RuntimeError("Not supported")
        else:
            return get_mt5_for_conditional_generation(model_name, from_pretrained=from_pretrained,
                                                      embedding_len=tokenizer_len)
    elif 't5' in model_name:
        if use_custom:
            return get_custom_t5_for_conditional_generation(model_name, tokenizer_len, from_pretrained=from_pretrained,
                                                            conf=conf, **kwargs)
        else:
            return get_t5_for_conditional_generation(model_name, from_pretrained=from_pretrained,
                                                     embedding_len=tokenizer_len)
    else:
        if use_custom:
            raise RuntimeError("Not supported")
        else:
            get_info_logger(__name__).warning('Unrecognised model name, using AutoModel API')
            return get_automodel_for_conditional_generation(model_name, from_pretrained=from_pretrained,
                                                            embedding_len=tokenizer_len)
