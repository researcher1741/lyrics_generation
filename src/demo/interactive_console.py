from copy import copy
from pprint import pprint
import omegaconf
import hydra

import pytorch_lightning as pl
import torch
import os

import logging

from transformers import LogitsProcessorList, StoppingCriteriaList
from src.lyrics_generation.generate import generate
from src.lyrics_generation.misc.reranker import SchemaScorer

import sys
from src.lyrics_generation_utils.utils import PhonemeFactory, VerseEndStopCriteria, get_info_logger, \
    set_pytorch_lightning_logger_level
from src.lyrics_generation_utils.init_utils import init_everything_for_evaluation
from src.lyrics_generation_utils.constants import END_LYRICS, RHYME_TOKENS, SCHEMA, SENTENCE_END, SEP, TAG_END

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def get_rhyming_word_and_rhyme_letter(sentence, use_sep=False):
    sentence = sentence.strip()
    rhyme_letter = sentence[:7]
    if use_sep:
        return sentence[7:sentence.index(SEP)], rhyme_letter
    if sentence.endswith(')'):
        if '(' in sentence:
            sentence = sentence[:sentence.rindex('(')]
    return sentence.split()[-1], rhyme_letter


def interactive_console(conf: omegaconf.DictConfig) -> None:
    generation_logger = get_info_logger('Generation Logger')
    if 'checkpoint_path' not in conf or 'tokenizer_path' not in conf.model:
        generation_logger.error('out_dir or checkpoint_path are not in conf, please pass it when running '
                                'the script as follows:\npython generate.py +checkpoint_path=/path/to/checkpoint '
                                '+model.tokenizer_path=/path/to/tokenizer')
        sys.exit(1)
    # reproducibility
    pl.seed_everything(conf.train.seed)
    set_pytorch_lightning_logger_level(logging.INFO)

    batch_size = conf.generation.batch_size
    conf.data.batch_size = batch_size
    _, pl_module, _, tokenizer = init_everything_for_evaluation(conf)
    # pl_module.force_schema = False
    device = 'cpu'
    if hasattr(conf, 'device'):
        device = conf.device
    pl_module.to(device)

    generation_args = dict(max_length=300,
                           num_beams=4,
                           no_repeat_ngram_size=2,
                           early_stopping=False,
                           do_sample=True,
                           decoder_start_token_id=tokenizer.bos_token_id,
                           # temperature=0.8,
                           # top_k = 10,
                           # length_penalty = 0.9,
                           forced_bos_token_id=None,
                           num_return_sequences=20
                           )
    pprint(generation_args)
    scorer = SchemaScorer()
    while True:
        in_str = '<title> Lower Body<artist> Chris Brown'
        schema = 'RHYME_A RHYME_B RHYME_C RHYME_B RHYME_A RHYME_B RHYME_A RHYME_B'
        letter_schema = [x.replace('RHYME_', '') for x in schema.split()]
        in_str = in_str + '<schema>' + schema
        input_ids = tokenizer.encode(in_str)
        print('=' * 50)
        print(tokenizer.decode(input_ids))
        input_ids = torch.LongTensor([input_ids] * 2)
        letter_schema = [letter_schema] * 2
        with torch.no_grad():
            generations = generate(pl_module, input_ids.to(device), letter_schema,
                                   batch={
                                       'decoder_rhyming_ids': None,
                                       'decoder_position_ids': None
                                   },
                                   **generation_args)
            generations = generations.view(input_ids.shape[0], -1, generations.shape[-1])
        for i, elem in enumerate(generations):

            all_decoded = [
                decoded.replace('<schema>', '<schema>\n').replace('<sentence_end>', '<sentence_end>\n').replace('<pad>',
                                                                                                                '')
                for decoded in tokenizer.batch_decode(elem)]
            all_scores = list()

            for decoded in all_decoded:
                decoded = decoded.strip()
                rhyming_words = list()
                generated_schema = list()
                for sentence in decoded.split('<sentence_end>'):
                    if len(sentence) == 0:
                        continue
                    rhyming_word, rhyming_letter = get_rhyming_word_and_rhyme_letter(sentence)
                    generated_schema.append(rhyming_letter)
                    rhyming_words.append(rhyming_word)
                score = scorer.score_candidate(rhyming_words, generated_schema)
                all_scores.append(score)
                # print(decoded)
                # print()
                # print('=' * 20)
            k = 1
            for decoded, score in sorted(zip(all_decoded, all_scores), key=lambda elem: elem[1], reverse=True):
                print(f'{k} - Score: {score:.3f}')
                print(decoded)
                print('=' * 20)
                print()
                k += 1
            print(f'END GENERATIONS FOR {i}')

        input()


def generate_sentence_by_sentence(model, input_ids, schemas, tokenizer, device, **generation_args):
    logger = logging.getLogger('transformers.generation_utils')
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    decoder_input_ids = input_ids + [tokenizer.encode(SCHEMA, add_special_tokens=False)[0]]
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    encoder = model.get_encoder()
    encoder_outputs = encoder(input_ids=input_ids, return_dict=True)
    sentence_end_id = tokenizer.encode(SENTENCE_END, add_special_tokens=False)[0]
    stop_criteria = VerseEndStopCriteria(sentence_end_id)
    stop_criteria_list = StoppingCriteriaList([stop_criteria])
    for schema in schemas:
        for i, letter in enumerate(schema):
            rhyme_token_id = tokenizer.encode(RHYME_TOKENS[ord(letter) - ord('A')], add_special_tokens=False)
            decoder_input_ids += rhyme_token_id
            decoder_input_ids = torch.LongTensor(decoder_input_ids).unsqueeze(0).to(device)
            generated_verse = model.generate(**generation_args,
                                             stopping_criteria=stop_criteria_list,
                                             decoder_input_ids=decoder_input_ids,
                                             encoder_outputs=copy(encoder_outputs))

            decoder_input_ids = generated_verse[0].tolist()[:-1] + [sentence_end_id]
    logger.setLevel(old_level)
    return decoder_input_ids


@hydra.main(config_path="../../conf", config_name="root_pretrain_t5")
def main(conf: omegaconf.DictConfig):
    interactive_console(conf)


if __name__ == "__main__":
    main()
