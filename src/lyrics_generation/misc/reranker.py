from typing import List
from src.lyrics_generation.evaluation.rhyming_evaluation2 import do_rhyme
from phonemizer.backend import BACKENDS
from src.lyrics_datasets.multilingual_processing.blockify_dataset import language2code


class SchemaScorer:
    def __init__(self, language) -> None:
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
        self.backends_dict = backends_dict

    def __build_rhyming_and_non_rhyming_pairs(self, schema):
        pairs = list()
        non_pairs = list()
        for i in range(len(schema)):
            for j in range(i + 1, len(schema), 1):
                i_letter = schema[i]
                j_letter = schema[j]
                if i_letter == j_letter:
                    pairs.append((i, j))
                else:
                    non_pairs.append((i, j))
        return pairs, non_pairs

    def score_candidate(self, candidate_rhyming_words: List[str], schema: List[str], gold_schema: List[str],
                        language: str):
        pairs, negative_pairs = self.__build_rhyming_and_non_rhyming_pairs(schema)
        schema_correct = 0
        for s, g in zip(schema, gold_schema):
            if s == g:
                schema_correct += 1
        schema_recall = schema_correct / len(gold_schema)
        true_positive = 0
        tot = 0
        for i, j in pairs:
            s1, s2 = candidate_rhyming_words[i], candidate_rhyming_words[j]
            if do_rhyme(s1, s2, self.backends_dict[language2code[language]], language):
                true_positive += 1
            tot += 1  # fp + tp
        true_negative = 0
        negative_tot = 0
        for i, j in negative_pairs:
            if not do_rhyme(candidate_rhyming_words[i], candidate_rhyming_words[j],
                            self.backends_dict[language2code[language]], language):
                true_negative += 1
            negative_tot += 1  # fn + tn
        precision = true_positive / max(tot, 1)
        npv = true_negative / max(negative_tot, 1)
        return precision * npv * schema_recall
        # we maximise the rhymes that were rightfully produced and the rhymes that were rightfully not produced
        # precision = 1.0 if all expected rhymes are produced according to the schema
        # npv = 1.0 if all sentences that should not rhyme (according to the schema) do not rhyme

    def score_candidates(self, all_candidates, all_schemas, gold_schemas, languages=None):
        scores = list()
        if languages is None:
            languages = [None] * len(all_candidates)
        for candidate, schema, gold_schema, language in zip(all_candidates, all_schemas, gold_schemas, languages):
            score = self.score_candidate(candidate, schema, gold_schema, language)
            scores.append(score)
        return scores
