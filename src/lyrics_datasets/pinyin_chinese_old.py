import re
import os
import sys
import string
import random
from typing import Dict, Union
import pronouncing
from pypinyin import pinyin, lazy_pinyin, Style

# rhyming rule reference: https://zhuanlan.zhihu.com/p/162004826 We adopt shisiyun (十四韵)
# We exclude some vowel combinations:'ueng' 'er'
rhyming_rule = {"group_1": ['ong', 'iong', 'eng', 'ing'],
                "group_2": ['ang', 'iang', 'uang'],
                'group_3': ['ei', 'ui'],
                'group_4': ['ai', 'uai'],
                'group_5': ['en', 'in', 'un'],
                'group_6': ["an", "uan", 'ian'],
                'group_7': ['ao', 'iao'],
                'group_8': ['o', 'uo', 'e'],
                'group_9': ['ve', 'ie', 'ue'],
                'group_10': ['a', 'ia', 'ua'],
                'group_11': ['ou', 'iu'],
                'group_12': ['i'],
                'group_13': ['v'],
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
