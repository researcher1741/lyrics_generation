import omegaconf
from omegaconf import OmegaConf
import hydra

import pytorch_lightning as pl
import torch
import jsonlines
import os
from tqdm import tqdm

import logging
from collections import deque

from transformers import AutoConfig
import sys
import re
from src.lyrics_generation.data_modules.simple_data_module import SimpleDataModule
from src.lyrics_generation.misc.reranker import SchemaScorer
from src.lyrics_generation.modules.encoder_decoder_module import EncoderDecoderModule
from src.lyrics_generation_utils.utils import get_info_logger, get_lyrics_tokenizer, set_pytorch_lightning_logger_level
# from src.lyrics_generation_utils.constants import BLOCK_END, END_LYRICS, RHYME_TOKENS, SCHEMA, SENTENCE_END, SEP, TAG_END
from src.lyrics_generation_utils.constants import BLOCK_END, END_LYRICS, SCHEMA, SENTENCE_END, SEP, TAG_END

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def run_generation(conf, pl_module, pl_data_module, tokenizer, language='english', **gen_args):
    gpus = conf.train.pl_trainer.gpus
    if gpus is not None and gpus != 0:
        if isinstance(gpus, omegaconf.ListConfig):
            gpus = gpus[0]
        else:
            raise RuntimeError('Cannot generate with gpus=', gpus)
    else:
        gpus = 'cpu'

    pl_module.to(gpus)
    pl_module.model.config.bos_token_id = gen_args.get('bos_token_id', tokenizer.bos_token_id)
    pl_module.model.config.eos_token_id = gen_args.get('eos_token_id', tokenizer.eos_token_id)
    eos_token = tokenizer.decode(pl_module.model.config.eos_token_id)
    model_conf = pl_module.model.config
    generation_args = dict(
        max_length=conf.generation.max_length if 'max_length' in conf.generation else 50,
        num_beams=conf.generation.num_beams if 'num_beams' in conf.generation else 5,
        no_repeat_ngram_size=conf.generation.no_repeat_ngram_size if 'generation_no_repeat_ngram_size' in conf.generation else 2,
        early_stopping=conf.generation.no_repeat_ngram_size if 'generation_no_repeat_ngram_size' in conf.generation else True,
        decoder_start_token_id=model_conf.decoder_start_token_id,
        # encode(RHYME_TOKENS[0], add_special_tokens=False)[0],
        do_sample=conf.generation.do_sample if 'do_sample' in conf.generation else False,
        forced_bos_token_id=None,
        num_return_sequences=conf.generation.num_return_sequences if 'num_return_sequences' in conf.generation else 20
    )
    if gen_args:
        generation_args.update(gen_args)
    out_dir = conf.out_dir
    out_dir += f'.ret_seq_{generation_args.get("num_return_sequences", 1)}.do_sample_{generation_args.get("do_sample", False)}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pl_data_module.setup(split_to_setup={'test'})
    use_sep = pl_data_module.data_version == '0.2.2' or pl_data_module.data_version == '0.2.1'
    test_loader = pl_data_module.generation_dataloader(task='lyrics_generation')
    gen_path = os.path.join(out_dir, 'generations.txt')
    gold_path = os.path.join(out_dir, 'reference.txt')

    os.system('clear')
    print("GENERATING WITH PARAMETERS:")

    print('Force Schema', pl_module.force_schema if hasattr(pl_module, 'force_schema') else False)
    print('Force SEP', pl_module.force_first_word_and_sep if hasattr(pl_module, 'force_first_word_and_sep') else False)
    print('Out path', gen_path)
    print('language', language)
    for k, v in generation_args.items():
        print(k, v)
    scorer = SchemaScorer(language=language)
    bar = tqdm(test_loader, desc='generating')
    i = 0
    with torch.no_grad():
        with open(gen_path, 'w') as generation_writer, open(gold_path, 'w') as gold_writer, \
                jsonlines.open(gen_path + '.jsonl', 'w') as gen_json_writer, jsonlines.open(gold_path + '.jsonl',
                                                                                            'w') as gold_json_writer:
            for batch in bar:
                if 'input_ids' not in batch and 'lyrics_generation' in batch:
                    batch = batch['lyrics_generation']
                batch = pl_data_module.transfer_batch_to_device(batch, gpus)
                schema = batch.get('schema', None)
                if 'decoder_input_ids' in batch and batch['decoder_input_ids'][0][0].item() != generation_args[
                    'decoder_start_token_id']:
                    print("WARN: detected wrong decoder_start_token_id, changing it from ",
                          generation_args['decoder_start_token_id'],
                          f'({tokenizer.decode(generation_args["decoder_start_token_id"])}) to',
                          batch['decoder_input_ids'][0][0].item(),
                          f'({tokenizer.decode([batch["decoder_input_ids"][0][0]])})')
                    generation_args['decoder_start_token_id'] = batch['decoder_input_ids'][0][0].item()
                generations = generate(pl_module, batch['input_ids'], schema, batch, **generation_args)
                # print_generations(tokenizer, batch['input_ids'], generation_logger, generations)
                if 'labels' in batch:
                    gold_ids = batch['labels']
                    gold_ids = torch.masked_fill(gold_ids, gold_ids == -100, tokenizer.pad_token_id)

                else:
                    gold_ids = None
                language_ids = batch.get('languages', None)
                languages = []
                if language_ids is not None:
                    if isinstance(language_ids[0], str):
                        languages = language_ids
                    else:
                        for ids in language_ids:
                            languages.append(tokenizer.decode(ids).split()[-1])
                else:
                    languages = None
                dump_for_evaluation(generation_writer, gold_writer,
                                    gen_json_writer, tokenizer,
                                    batch['input_ids'],
                                    generations, gold_ids, scorer, use_sep,
                                    language=language,
                                    elem_languages=languages,
                                    eos_token=eos_token)


def test(conf: omegaconf.DictConfig) -> None:
    generation_logger = get_info_logger('Generation Logger')
    if 'checkpoint_path' not in conf and 'out_dir' not in conf:
        generation_logger.error('out_dir or checkpoint_path are not in conf, please pass it when running the script as follows:\n\
        python generate.py +checkpoint_path=/path/to/checkpoint +out_dir=/path/to/outdir/ [model.force_schema=[True,False]')
        sys.exit(1)

    # reproducibility
    pl.seed_everything(conf.train.seed)
    set_pytorch_lightning_logger_level(logging.INFO)
    current_conf_path = '/tmp/hparams.yaml'
    with open(current_conf_path, 'w') as writer:
        writer.write(OmegaConf.to_yaml(conf))
    tokenizer = get_lyrics_tokenizer(conf)

    pl_module = EncoderDecoderModule.load_from_checkpoint(conf.checkpoint_path, hparams_file=current_conf_path,
                                                          map_location='cpu', strict=True)
    pl_data_module = SimpleDataModule(tokenizer, conf)

    conf.data.test_batch_size = conf.generation.batch_size
    run_generation(conf, pl_module, pl_data_module, tokenizer, language=conf.data.language)


def get_rhyming_word_and_rhyme_letter(sentence, use_sep=False):
    sentence = sentence.strip()
    rhyme_letter = sentence[:7]
    if use_sep:
        if SEP in sentence:
            return sentence[7:sentence.index(SEP)].strip(), rhyme_letter
        else:
            return '', rhyme_letter
    if sentence.endswith(')'):
        if '(' in sentence:
            sentence = sentence[:sentence.rindex('(')]
    return sentence.split()[-1], rhyme_letter


def get_rhyming_words_and_schemas(batch_gens, use_sep=False):
    all_schemas = []
    all_rhyming_words = []
    for batch_elem in batch_gens:
        schema = []
        rhyming_words = []
        for sentence in batch_elem.split(SENTENCE_END):
            if len(sentence) == 0:
                continue
            word, schema_l = get_rhyming_word_and_rhyme_letter(sentence, use_sep)
            schema.append(schema_l)
            rhyming_words.append(word)
        all_schemas.append(schema)
        all_rhyming_words.append(rhyming_words)
    return all_rhyming_words, all_schemas


def dump_for_evaluation(generation_writer, gold_writer,
                        generation_json_writer, tokenizer, inputs,
                        generations, gold_ids, scorer, use_sep,
                        language='english',
                        elem_languages=None,
                        eos_token=None):
    if eos_token is None:
        eos_token = tokenizer.eos_token
    decoded_inp = [tokenizer.decode(inp).replace(tokenizer.pad_token, '').strip() for inp in inputs]
    decoded_gens = tokenizer.batch_decode(generations)
    if elem_languages is None:
        elem_languages = [language] * len(inputs)
    for i in range(len(decoded_gens)):
        dg = decoded_gens[i]
        dg = dg.replace(END_LYRICS, '').replace(tokenizer.pad_token, '').strip()
        if eos_token is not None:
            dg = dg.replace(eos_token, '')
        decoded_gens[i] = dg
    input_schemas = []
    extended_langs = []
    samples_per_elem = generations.shape[0] // inputs.shape[0]
    for inp, lang in zip(inputs, elem_languages):
        aux = deque()
        for i in range(len(inp) - 1, 0, -1):
            item = inp[i].item()
            if item == tokenizer.pad_token_id:
                continue
            elif item == tokenizer.encode(SCHEMA, add_special_tokens=False):
                break
            else:
                item_str = tokenizer.decode(inp[i])
                if item_str.startswith('RHYME_'):
                    aux.appendleft(item_str)
        input_schemas.extend([list(aux)] * samples_per_elem)
        extended_langs.extend([lang] * samples_per_elem)
    if gold_ids is not None:
        decoded_golds = [tokenizer.decode(g).replace(END_LYRICS, '').replace(tokenizer.pad_token, '').strip() for g in
                         gold_ids]
    else:
        decoded_golds = [None] * len(decoded_gens)
    top_gens = list()
    top_schemas = list()
    if samples_per_elem > 1:
        for i in range(0, len(decoded_gens), samples_per_elem):
            batch_gens = decoded_gens[i:i + samples_per_elem]
            batch_schemas = input_schemas[i:i + samples_per_elem]
            langs = extended_langs[i:i + samples_per_elem]
            rhyming_words, rhyming_schemas = get_rhyming_words_and_schemas(batch_gens, use_sep)
            scores = scorer.score_candidates(rhyming_words, rhyming_schemas, batch_schemas, languages=langs)

            top_gen = sorted(zip(batch_gens, scores), reverse=True, key=lambda elem: elem[1])[0][0]
            top_schemas.append(batch_schemas[0])
            top_gens.append(top_gen)
    else:
        top_gens = decoded_gens
        top_schemas = input_schemas

    generations = []
    rhyming_schemas = []
    for inp, gen, gold, schema, lang in zip(decoded_inp, top_gens, decoded_golds, top_schemas, elem_languages):
        gen = gen.replace(inp, '').replace(tokenizer.pad_token, '').replace('<pad>', '').replace(SENTENCE_END,
                                                                                                 SENTENCE_END + '\n').replace(
            TAG_END, TAG_END + '\n')
        gen = gen.replace(BLOCK_END, '\n' + BLOCK_END + '\n')
        inp = inp.replace(tokenizer.pad_token, '').replace('<pad>', '').replace(SENTENCE_END,
                                                                                SENTENCE_END + '\n').replace(TAG_END,
                                                                                                             TAG_END + '\n')
        if gold:
            gold = gold.replace(tokenizer.pad_token, '').replace('<pad>', '').replace(SENTENCE_END,
                                                                                      SENTENCE_END + '\n').replace(
                TAG_END, TAG_END + '\n')
            gold_writer.write(inp + '\n' + gold + '\n\n')
            gold_writer.flush()
        if lang == 'chinese':
            inp = inp.replace(' ', '')
            gen = gen.replace(' ', '')

        generations.append(gen)
        generation_writer.write(inp + '\n' + gen + '\n\n')
        generation_writer.flush()
        generated_schema = re.findall(r'RHYME_[A-Z]', gen)
        if schema is not None and generated_schema != schema:
            print('[WARNING] generated schema is different from input schema!')
        generation_json_writer.write(
            {'generation': gen, 'input': inp, 'gold': gold, 'rhyming_schema': schema, 'language': lang})


def print_generations(tokenizer, in_data, logger, generations):
    in_data = [tokenizer.decode(data) for data in in_data]
    all_generations = [tokenizer.decode(gen) for gen in generations]

    for data, generation in zip(in_data, all_generations):
        logger.info('Prompt:')
        logger.info(data)
        logger.info('Generation')
        logger.info(generation.replace(data, ''))
        logger.info('=' * 40)


def generate(model, in_data, schema, batch=None, **generation_args):
    logger = logging.getLogger('transformers.generation_utils')
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    in_dict = {'in_data': in_data}
    in_dict.update(generation_args)
    if schema is not None:
        in_dict['schema'] = schema
    generations = model.generate(**in_dict, batch=batch)
    logger.setLevel(old_level)

    return generations


@hydra.main(config_path="../../../conf", config_name="root_t5.yaml")
def main(conf: omegaconf.DictConfig):
    test(conf)


if __name__ == "__main__":
    main()
