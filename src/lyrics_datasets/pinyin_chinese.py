import re
import os
import sys
import string
import random
from typing import Dict, Union
import pronouncing
from pypinyin import pinyin, lazy_pinyin, Style

## new 14 rhyming groups
rhyming_rule = {'group_1': ['a', 'ia', 'ua'],
                'group_2': ['o', 'e', 'uo'],
                'group_3': ['ie', 've'],
                'group_4': ['ai', 'uai'],
                'group_5': ['ei', 'ui', 'uei'],
                'group_6': ['ao', 'iao'],
                'group_7': ['ou', 'iu', 'iou'],
                'group_8': ['an', 'ian', 'uan', 'van'],
                'group_9': ['en', 'in', 'ien', 'un', 'uen', 'vn'],
                'group_10': ['ang', 'iang', 'uang'],
                'group_11': ['eng', 'ing', 'ieng', 'ong', 'ueng', 'iong'],
                'group_12': ['i', 'er', 'v'],  # v write as u when meeting j q x, but still pronounce as v
                'group_13': ['zhi', 'chi', 'shi', 'zi', 'ci', 'si', 'ri'],  # -i: no finals
                'group_14': ['u']}
# Produce a reverse library
classification = {}

for k in rhyming_rule:
    for items in rhyming_rule[k]:
        classification[items] = k

cache = {}


class Pinyin():
    def __init__(self, word):
        if word not in cache:
            self.word = word
            self.vowels = pinyin(word, style=Style.FINALS, strict=False)
            self.rhyming_group = classification.get(self.vowels[0][0], None)
            cache[word] = (self.word, self.vowels, self.rhyming_group)
        else:
            self.word, self.vowels, self.rhyming_group = cache[word]


def rhyme_zh(word1, word2):
    word1 = Pinyin(word1)
    word2 = Pinyin(word2)
    if word1.rhyming_group is not None and word2.rhyming_group is not None and word1.rhyming_group == word2.rhyming_group:
        return True
    else:
        return False
