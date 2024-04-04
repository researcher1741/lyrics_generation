import logging
import sys
from typing import Dict, List
from src.lyrics_generation_utils.constants import *
from transformers import AutoTokenizer, LogitsProcessor
from tokenizers.processors import TemplateProcessing
from src.lyrics_datasets.phonetics import near_rhyme, perfect_rhyme
from phonemizer.backend import EspeakBackend, FestivalBackend
from phonemizer.separator import Separator
from phonemizer.punctuation import Punctuation
import os
from src.lyrics_generation_utils.constants import LYRICS_SPECIAL_TOKENS
import torch

if not hasattr(torch, 'inf'):
    import math

    torch.inf = math.inf


class SchemaEnforcingLogitsProcessor(LogitsProcessor):
    def __init__(self, schema, sentence_end_id, tokenizer, trigger_after_schema_token=False, sample=False):
        self.schema = schema
        self.sentence_end_id = sentence_end_id
        self.tokenizer = tokenizer
        self.schema_letter_ids = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in RHYME_TOKENS]
        self.eos_token_id = tokenizer.eos_token_id
        self.schema_token_id = tokenizer.encode(SCHEMA, add_special_tokens=False)[0]
        self.trigger_after_schema_token = trigger_after_schema_token
        if not sample:
            self.min_val = -1000
            self.max_val = 0.0
        else:
            self.min_val = -1000
            self.max_val = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_sentences_per_example = torch.sum(input_ids == self.sentence_end_id, -1)
        schema_triggers = torch.sum(input_ids == self.schema_token_id, -1)
        last_generations = input_ids[:, -1]
        new_scores = list()
        if input_ids.shape[1] == 1:
            rhyme_token = RHYME_TOKENS[0]
            rhyme_token_id = self.tokenizer.encode(rhyme_token, add_special_tokens=False)[0]
            mask = torch.ones_like(scores).bool()
            mask[:, rhyme_token_id] = False
            scores.masked_fill_(mask, self.min_val)
            scores[:, rhyme_token_id] = self.max_val
            return scores

        for i, (lg, next_token_scores, schema, trigger, num_sentences) in enumerate(
                zip(last_generations, scores, self.schema, schema_triggers, num_sentences_per_example)):
            if num_sentences.item() < len(schema):
                next_token_scores[self.eos_token_id] = self.min_val  # minimise chances of predicting eos
                next_token_scores[
                    self.schema_letter_ids] = self.min_val  # minimise chances of predicting rhyme_tokens in the middle of a sentence

            if lg.item() == self.sentence_end_id:
                if num_sentences.item() >= len(schema):  # generate end of sentence.
                    # check number of sentences generated
                    # check if rhyming schema is over (then modify next_token_scores to output eos_token_id)
                    mask = torch.ones_like(next_token_scores).bool()
                    mask[self.eos_token_id] = False
                    next_token_scores.masked_fill_(mask, self.min_val)
                    next_token_scores[self.eos_token_id] = self.max_val
                else:  # generate next rhyming token
                    schema_letter = schema[num_sentences]
                    # otherwise, get the next rhyming letter (A,B, etc.)
                    rhyme_token = RHYME_TOKENS[ord(schema_letter) - ord('A')]
                    # compute the id for it
                    rhyme_token_id = self.tokenizer.encode(rhyme_token, add_special_tokens=False)[0]
                    # modify next_token_scores to output that token_id.
                    mask = torch.ones_like(next_token_scores)
                    mask[rhyme_token_id] = 0
                    next_token_scores.masked_fill_(mask.bool(), self.min_val)
                    next_token_scores[rhyme_token_id] = self.max_val

            # elif self.tokenizer.decode(lg.item()) in RHYME_TOKENS: # add space after RHYME_TOKEN
            #     mask = torch.ones_like(next_token_scores)
            #     token = self.tokenizer.tokenize(' .')[0]
            #     token = token.replace('.', '')
            #     token_id = self.tokenizer.convert_tokens_to_ids(token)
            #     mask[token_id] = False
            #     next_token_scores.masked_fill_(mask.bool(), self.min_val)
            #     next_token_scores[token_id] = self.max_val

            new_scores.append(next_token_scores)
        return torch.stack(new_scores, 0)


class FirstWordSEPLogitProcessor(LogitsProcessor):
    def __init__(self, sentence_end_id, sep_token_id, tokenizer, sample=False):
        self.sentence_end_id = sentence_end_id
        self.sep_token_id = sep_token_id
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.space_token_id = self.tokenizer.encode(' .', add_special_tokens=False)[0]
        self.min_val = -torch.inf
        self.max_val = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) < 3:
            scores[:, self.sep_token_id] = self.min_val
            scores[:, self.sentence_end_id] = self.min_val
            scores[:, self.space_token_id] = self.min_val
            return scores
        start_id = input_ids[0, 0]
        ## TODO limitation: if we have a word that is split in multiple subwords we are truncating it.
        for i, is_sentence_end in enumerate(
                (input_ids[:, -3] == self.sentence_end_id) + (input_ids[:, -3] == start_id)):
            if is_sentence_end.item():
                scores[i, :] = self.min_val
                scores[i, self.sep_token_id] = self.max_val

        prev_rhyme_token_mask = torch.Tensor(
            [self.tokenizer.decode(x) in RHYME_TOKENS for x in input_ids[:, -1]]).bool()
        if sum(prev_rhyme_token_mask) > 0:
            scores[prev_rhyme_token_mask, self.sentence_end_id] = self.min_val
            scores[prev_rhyme_token_mask, self.space_token_id] = self.min_val
        return scores


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_info_logger(name=None, formatter=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.parent is not None:
        return logger

    if formatter is None:
        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


logger = get_info_logger(__name__)


def set_pytorch_lightning_logger_level(level):
    logger = logging.getLogger("pytorch_lightning")
    logger.setLevel(level)
    fh = logging.FileHandler("train.log")
    fh.setLevel(level)
    logger.addHandler(fh)


def get_custom_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.post_processor = TemplateProcessing(
        single="$A </s>",
        special_tokens=[
            ("</s>", tokenizer.encode('</s>')[0]),
        ]
    )
    return tokenizer


def get_lyrics_tokenizer(conf_or_name, add_lyrics_tokens=True):
    if isinstance(conf_or_name, str):
        model_name = conf_or_name
    else:
        conf = conf_or_name
        if "tokenizer_path" in conf.model:
            print('TOKENIZER_PATH')
            tokenizer_path = conf.model.tokenizer_path
            print(tokenizer_path)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f'Tokenizer loaded from pretrained files:\n{tokenizer_path}')
            if tokenizer.eos_token_id is None and tokenizer.sep_token_id is not None:
                tokenizer.eos_token = tokenizer.sep_token
            return tokenizer
        elif 'tokenizer_name' in conf.model:
            model_name = conf.model.tokenizer_name
        elif "pretrained_model_name_or_path" in conf.model:
            model_name = conf.model.pretrained_model_name_or_path
        else:
            model_name = conf.model.pretrained_model_name
    print(" ###  model_name: ", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f'Tokenizer loaded from model name: {model_name}')
    if add_lyrics_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': LYRICS_SPECIAL_TOKENS})
        logger.info(f'Tokenizer updated with the following tokens\n{LYRICS_SPECIAL_TOKENS}')
    if tokenizer.eos_token_id is None and tokenizer.sep_token_id is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def print_generations(tokenizer, in_data, logger, generations):
    in_data = [tokenizer.decode(data) for data in in_data]
    all_generations = [tokenizer.decode(gen) for gen in generations]

    for data, generation in zip(in_data, all_generations):
        logger.info('Prompt:')
        logger.info(data)
        logger.info('Generation')
        logger.info(generation.replace(data, ''))
        logger.info('=' * 40)


def generate(model, in_data, **generation_args):
    logger = logging.getLogger('transformers.generation_utils')
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    generations = model.model.generate(inputs=in_data,
                                       **generation_args
                                       )
    logger.setLevel(old_level)

    return generations


def remove_second_voice(line):
    if line.strip().endswith(')') and '(' in line:
        return line[:line.rindex('(')].strip()
    return line


def do_rhyme(w1: Dict, w2: Dict, use_similar_words=False):
    if perfect_rhyme(w1, w2):
        return True
    return use_similar_words and near_rhyme(w1, w2)


def count_syllables(lines):
    syllables_per_line = [[x for w in remove_second_voice(line).split() for x in w.split('|')] for line in lines]
    if all(len(l) == 0 for l in syllables_per_line):
        syllables_per_line = [[x for w in line.split() for x in w.split('|')] for line in lines]
    num_syllables_per_line = [len(l) for l in syllables_per_line if len(l) > 0]
    return num_syllables_per_line


def infer_rhyming_schema(text, phonemes_espeak, phonemes_festival, sentence_window=4, use_similar_to=True):
    lyrics = text.split('\n')
    lyrics = [remove_second_voice(l) for l in
              lyrics]  # sometimes verses end with text in parenthes (which indicate the second voice). This does not count for rhyming
    phonemes_espeak = [remove_second_voice(l) for l in phonemes_espeak.split('\n')]
    phonemes_festival = [remove_second_voice(l) for l in phonemes_festival.split('\n')]
    rhymes = [None] * len(lyrics)
    for i in range(0, len(lyrics)):
        current_sentence = lyrics[i].split()
        current_phonemes_espeak = phonemes_espeak[i].split()
        current_phonemes_festival = phonemes_festival[i].split()
        if len(current_sentence) == 0 or len(current_phonemes_espeak) == 0 or len(current_phonemes_festival) == 0:
            continue
        current_cs_token, current_e_token, current_f_token = current_sentence[-1], current_phonemes_espeak[-1], \
                                                             current_phonemes_festival[-1]
        current_last_word = {'w': current_cs_token, 'e': current_e_token, 'f': current_f_token}
        # current_interest_words = list(zip(a,b,c))
        start = i + 1
        end = start + sentence_window
        next_token_window, next_espeak_window, next_festival_window = lyrics[start:end], phonemes_espeak[
                                                                                         start:end], phonemes_festival[
                                                                                                     start:end]
        for (j, (next_sentence_l, next_phonemes_espeak, next_phonemes_festival)) in enumerate(
                zip(next_token_window, next_espeak_window, next_festival_window)):
            next_sentence = next_sentence_l.split()
            next_phonemes_espeak = next_phonemes_espeak.split()
            next_phonemes_festival = next_phonemes_festival.split()
            if len(next_sentence) == 0 or len(next_phonemes_espeak) == 0 or len(next_phonemes_festival) == 0:
                continue
            next_last_token, next_e, next_f = next_sentence[-1], next_phonemes_espeak[-1], next_phonemes_festival[-1]
            s = 1
            next_word = {'w': next_last_token, 'e': next_e, 'f': next_f}
            if lyrics[i] != next_sentence_l and do_rhyme(current_last_word, next_word, use_similar_to):
                if rhymes[i] is None:
                    rhymes[i] = list()
                rhymes[i].append(i + 1 + j)
                # break
    schema = [None] * len(lyrics)
    letter = 'A'
    for i, rr in enumerate(rhymes):
        if schema[i] is None:
            schema[i] = letter
            letter = chr(ord(letter) + 1)
        if rr is not None:
            for r in rr:
                if schema[r] is None:
                    schema[r] = schema[i]
            # letter = chr(ord(letter) + 1)

    return schema


class PhonemeFactory():
    def __init__(self, backends: List[str], language,
                 punctuation_marks=Punctuation.default_marks(),
                 preserve_punctuation=False,
                 with_stress=False) -> None:
        b2phonemizer = dict()
        for b in backends:
            if b == 'espeak':
                backend = EspeakBackend(language, punctuation_marks, preserve_punctuation, with_stress)
            elif b == 'festival':
                backend = FestivalBackend(language, punctuation_marks, preserve_punctuation)
            b2phonemizer[b] = backend

        self.b2phonemizer = b2phonemizer
        import logging
        from logging import getLogger
        self.logger = getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def phonemize(self, text: List[str], backend: str, separator: Separator):
        from phonemizer.phonemize import _phonemize
        if backend not in self.b2phonemizer.keys():
            self.logger.error('Unknown backend', backend)
            return None
        phonemizer = self.b2phonemizer[backend]
        return _phonemize(phonemizer, text, separator, strip=False, njobs=1, prepend_text=False)


def get_phonetics(texts, phoneme_factory: PhonemeFactory, espeak_separator=None, festival_seapartor=None):
    if espeak_separator is None:
        espeak_separator = Separator(phone='.', syllable='', word=' ')
    if festival_seapartor is None:
        festival_seapartor = Separator(phone='.', syllable='|', word=' ')
    try:
        batch_size = 500  # empirically, lists longer than this make phonemize crashing due to recursion error
        espeak = []
        festival = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            b_espeak = phoneme_factory.phonemize(batch,
                                                 'espeak',
                                                 separator=espeak_separator
                                                 )

            b_festival = phoneme_factory.phonemize(batch,
                                                   'festival',
                                                   separator=festival_seapartor
                                                   )
            espeak.extend(b_espeak)
            festival.extend(b_festival)

    except (RuntimeError, RecursionError):
        return None, None
    return espeak, festival


def get_phoneme_tokenizer(model_name,
                          tokenizer_folder='/lyrics_generation/data/genius/phonemised_dataset'):
    model_name = model_name.split('/')[-1]
    path = os.path.join(tokenizer_folder, model_name + '_tokenizer')
    return AutoTokenizer.from_pretrained(path)


from transformers import StoppingCriteria
import torch


class VerseEndStopCriteria(StoppingCriteria):
    def __init__(self, sentence_end_id):
        self.sentence_end_id = sentence_end_id

    def __call__(self, input_ids, *args, **kwds) -> bool:
        return torch.sum(input_ids[:, -1] == self.sentence_end_id) > 0
