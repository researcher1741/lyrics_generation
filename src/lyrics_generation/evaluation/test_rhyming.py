from src.lyrics_generation.evaluation.rhyming_evaluation2 import do_rhyme
from phonemizer.backend import BACKENDS

sentence_1 = '有时候我觉得自己像一只小小鸟'
sentence_2 = '想要飞却怎么样也飞不高'
phonemizer = BACKENDS['espeak'](
                'cmn',
                with_stress=True
            )

print(sentence_1, sentence_2, do_rhyme(sentence_1, sentence_2, phonemizer, 'chinese'))