from argparse import ArgumentParser
from src.lyrics_datasets.multilingual_processing.blockify_dataset import language2code
from tqdm import tqdm
import jsonlines
from src.lyrics_generation_utils.constants import SENTENCE_END
from phonemizer.backend import BACKENDS
import re
import editdistance
from src.lyrics_datasets.phonetics import Phonetic, near_rhyme


def evaluate(generation_path, language=None):
    with jsonlines.open(generation_path) as lines:
        eval_dict = rhyming_evaluation(lines, language)
        keys = ['micro_precision', 'macro_precision', 'schema_precision', 'micro_fp_rate', 'macro_fp_rate']
        print('\t'.join(keys))
        print('\t'.join([f'{eval_dict[x]:.4f}' for x in keys]))
        return eval_dict


def compute_fp_rate(sentences, schema, language, phoneme_factory):
    fp = 0
    tot = 0
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences[i + 1:]):
            real_j = i + j
            if schema[i] != schema[real_j]:
                if do_rhyme(s1, s2, phoneme_factory, language):
                    fp += 1
                tot += 1
    return fp, tot, fp / max(tot, 1)


def rhyming_evaluation(generations, language=None):
    num_support = 0
    correct = 0
    tot = 0
    correct_for_macro = []
    schema_correct = 0
    distances = 0
    schema_tot = 0
    schema_lens = 0
    gold_schema_lens = 0
    fp = 0
    fp_tot = 0
    macro_fp_rate = 0
    backends_dict = dict()
    if language is None or language == 'multi':
        for lang in language2code.values():
            backends_dict[lang] = BACKENDS['espeak'](
                lang,
                with_stress=True
            )
    else:
        backends_dict[language2code[language]] = BACKENDS['espeak'](
            language2code[language],
            with_stress=True
        )
    for item in tqdm(generations, desc='evaluating'):
        if '<lang>' in item['input']:
            item_lang = re.findall(r'<lang> .*<schema>', item['input'])[0].replace('<lang>', '').replace('<schema>',
                                                                                                         '').strip()
        if language != 'multi' and language is not None and 'language' in item and item_lang != language:
            continue
        phonemizer = backends_dict[language2code[item_lang]]
        generation = item['generation'].replace('<s>', '').replace('</s>', '')
        sentences = [s.strip() for s in generation.split(SENTENCE_END) if len(s.strip()) > 0]
        schema = [re.findall(r'RHYME_[A-Z]', s) for s in sentences]
        schema = [x[0] if len(x) > 0 else "None" for x in schema]
        local_correct = 0
        local_tot = 0
        rhyming_schema = item['rhyming_schema']
        if isinstance(rhyming_schema, str):
            rhyming_schema = rhyming_schema.split()

        dist = editdistance.eval(schema, rhyming_schema)
        distances += dist
        if ''.join(schema) == ''.join(rhyming_schema):
            schema_correct += 1
        gold_schema_lens += len(rhyming_schema)
        schema_lens += len(schema)
        schema_tot += 1
        for i, l1 in enumerate(schema):
            for j in range(i + 1, len(schema)):
                l2 = schema[j]
                if l1 == l2:
                    if do_rhyme(sentences[i], sentences[j], phonemizer, item_lang):
                        correct += 1
                        local_correct += 1
                    tot += 1
                    local_tot += 1
        correct_for_macro.append(local_correct / max(local_tot, 1))
        local_fp, local_tot, local_fp_rate = compute_fp_rate(sentences, schema, item_lang, phonemizer)
        fp += local_fp
        fp_tot += local_tot
        macro_fp_rate += local_fp_rate
        num_support += 1
    macro_precision = sum(correct_for_macro) / max(len(correct_for_macro), 1)
    micro_precision = correct / tot
    macro_fp_rate = macro_fp_rate / max(len(correct_for_macro), 1)
    micro_fp_rate = fp / fp_tot
    return {'Support': num_support, 'micro_precision': micro_precision, 'macro_precision': macro_precision,
            'micro_fp_rate': micro_fp_rate, 'macro_fp_rate': macro_fp_rate,
            'schema_precision': schema_correct / schema_tot}


def do_rhyme(sentence_1, sentence_2, phonemizer, language):
    sentence_1 = sentence_1.strip()
    sentence_2 = sentence_2.strip()
    if len(sentence_1) == 0 or len(sentence_2) == 0:
        return False
    if re.match(r'.*[.!?]$', sentence_1) is not None:
        sentence_1 = sentence_1[:-1]
    if re.match(r'.*[.!?]$', sentence_2) is not None:
        sentence_2 = sentence_2[:-1]
    if sentence_1.endswith(')') and '(' in sentence_1:
        sentence_1 = sentence_1[:sentence_1.rfind('(')]
    if sentence_2.endswith(')') and '(' in sentence_2:
        sentence_2 = sentence_2[:sentence_2.rfind('(')]
    word_1 = sentence_1.split()
    word_2 = sentence_2.split()
    if len(word_1) == 0 or len(word_2) == 0:
        return False
    word_1 = word_1[-1].lower()
    word_2 = word_2[-1].lower()
    word_1 = re.sub(r'[,.;\'":\[\]}{=\+-_\)\(\*&^%$#!`@]+', '', word_1)
    word_2 = re.sub(r'[,.;\'":\[\]}{=\+-_\)\(\*&^%$#!`@]+', '', word_2)
    if word_1.lower() == word_2.lower():
        return True
    p1 = Phonetic(word_1, phonemizer)
    p2 = Phonetic(word_2, phonemizer)
    return near_rhyme(p1, p2, language)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path',
                        default='./LG/GENERATIONS/T5_multilingual/epoch=4-step=349637.force_schema_True.ret_seq_20.do_sample_True/generations.txt.jsonl')
    parser.add_argument('--language', default='multi')
    args = parser.parse_args()
    evaluate(args.path, language=args.language)
