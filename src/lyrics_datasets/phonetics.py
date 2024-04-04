import re
import os
import sys
import string
import random
from typing import Dict, Union
import pronouncing
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator
from phonemizer.backend import BACKENDS
from pypinyin import pinyin, lazy_pinyin, Style

IPA_VOWELS = {'i', 'ɪ', 'e', 'ɛ', 'æ', 'a', 'ʌ', 'ə', 'u', 'ʊ', 'o', 'ɔ'}


# VOWELS_BY_LANGUAGE = {'english': {'a', 'e', 'i', 'o', 'u', 'y'},
#                       'italian': {'a', 'e', 'i', 'o', 'u', 'y'}, 
#                       'french':{'a', 'e', 'i', 'o', 'u', 'y'}, 
#                       'croatian':{'a', 'e', 'i', 'o', 'u'},
#                       'danish':{'a', 'e', 'i', 'o', 'u', 'ø', 'y', 'æ', 'å'},
#                       'dutch':{'a', 'e', 'i', 'o', 'u', 'y'},
#                       'finnish':{'a', 'e', 'i', 'o', 'u', 'æ', 'ø'},
#                       'german':{'a', 'e', 'i', 'o', 'u', 'ä', 'ö', 'ü'},
#                       'hausa':{'a', 'e', 'i', 'o', 'u'},
#                       'hungarian':{'a', 'e', 'i', 'o', 'ö', 'u', 'ü'},
#                       'indonesian':{'a', 'e', 'i', 'o', 'u'},
#                       'japanese':{'あ', 'い', 'う', 'え', 'お'},
#                       'lithuanian':{'a', 'e', 'i', 'o', 'u'},
#                       'norwegian':{'a', 'e', 'i', 'o', 'u', 'ø', 'y', 'æ', 'å'},
#                       'polish':{"a", "e", "i", "o", "u", "ó", "y", "ę", "ą"},
#                       'portuguese':{'a', 'e', 'i', 'o', 'u'},
#                       'slovak':{'a', 'e', 'i', 'o', 'u', 'y', 'á', 'ä', 'é', 'í', 'ó', 'ó', 'ú', 'ý'},
#                       'spanish':{'a', 'e', 'i', 'o', 'u', 'y'},
#                       'swedish':{'a', 'e', 'i', 'o', 'u', 'y', 'ä', 'ö', 'å'},
#                       'turkish':{'a', 'e', 'i', 'o', 'u', 'ö', 'ü', 'ı'}
#                       }

# Is a string a word?
def is_word(str):
    str_set = set(str)
    return len(str) > 0  # and str_set.issubset(STR_LETTERS)


# Is a letter a vowel?
def is_vowel(letter, language):
    return letter in IPA_VOWELS  # VOWELS_BY_LANGUAGE[language]


# Is a letter a consonant?
def is_consonant(letter):
    return not is_vowel(letter)


# Is a phoneme a vowel sound?
# Pass in a phoneme (get this from Phonetic.phones_list)
def is_vowel_sound(phoneme, language):
    return len(phoneme) > 0 and is_vowel(phoneme[0],
                                         language)  # and (is_vowel(phoneme[-1]) or phoneme[-1] in ['h', 'x', 'y'])


# Is a phoneme a consonant sound?
# Pass in a phoneme (get this from Phonetic.phones_list)
def is_consonant_sound(phoneme):
    return not is_vowel_sound(phoneme)


# Return the number of vowel phonemes in a phoneneme list.
# Pass in a list of phonemes (get this from Phonetic.phones_list)
def num_vowel_phones(phone_list, language):
    count = 0
    for ph in phone_list:
        if len(ph) > 0 and is_vowel(ph[0], language):
            count = count + 1
    return count


# Return the indicies of a character (ch) in a string (s)
def find_char_indexes(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


# Store phoneme information here for fast lookup
PHONE_CACHE = {}  # Store results in cache for faster lookup


# Phonetic class
# word: orignal word
class Phonetic():
    def __init__(self, word, phonemizer, festival=None, espeak=None):
        global PHONE_CACHE
        self.word = word  # Remember the word
        # festival = festival                   # festival results
        self.phonemizer = phonemizer
        # espeak = espeak                     # espeak results
        # If the word is in cache, use those results
        if espeak is None:
            if word in PHONE_CACHE:
                espeak = PHONE_CACHE[word]
            else:
                # API calls
                # festival = phonemize(word, separator=Separator(phone='.', syllable='|', word=' ')).strip()
                espeak = os.linesep.join(
                    phonemizer.phonemize([word], separator=Separator(phone='.', syllable='', word=' '), strip=False))
                # espeak = phonemize(word, backend='espeak', with_stress=True, separator=Separator(phone='.', syllable='', word=' ')).strip()
                # Store the results
                PHONE_CACHE[word] = espeak
        else:
            PHONE_CACHE[word] = espeak
        # List of syllables. Each syllable is a list of phones
        # self.syllables_with_phones = [syl[:-1].split('.') for syl in festival.split('|')[:-1]]
        # # Just the syllables in raw form
        # self.syllables = festival.replace('.', '')
        # # Just the syllables in list form
        # self.syllables_list = self.syllables.split('|')[:-1]
        # # Just the phones in raw form
        # self.z = festival.replace('|', '')
        # Just the phones in list form
        self.phones_list = espeak.split('.')[:-1]
        self.major_stress = 0  # The major stresses phone (index)
        self.minor_stresses = []  # List of minor stressed phones (list of indices)
        # Compute the major stress
        major_stress_char_idxs = find_char_indexes(espeak, 'ˈ')
        if len(major_stress_char_idxs) > 0:
            stress_idx = espeak[:major_stress_char_idxs[0]].count('.')
            if stress_idx < len(self.phones_list):
                self.major_stress = stress_idx
            else:
                # Find the next earliest vowel phone
                for i in range(len(self.phones_list)):
                    if is_vowel_sound(self.phones_list[len(self.phones_list) - i - 1]):
                        self.major_stress = len(self.phones_list) - i - 1
                        break
        # compute the minor stresses
        minor_stress_char_idxs = find_char_indexes(espeak, 'ˌ')
        if len(minor_stress_char_idxs) > 0:
            for char_idx in minor_stress_char_idxs:
                stress_idx = espeak[:char_idx].count('.')
                if stress_idx < len(self.phones_list):
                    self.minor_stresses.append(stress_idx)
                else:
                    # Find the next earliest vowel phone
                    for i in range(len(self.phones_list)):
                        if is_vowel_sound(self.phones_list[len(self.phones_list) - i - 1]):
                            self.minor_stresses.append(len(self.phones_list) - i - 1)
                            break
            # all stressed phones (list of indices)
        self.all_stresses = sorted([self.major_stress] + self.minor_stresses[:])

    def num_phones(self):
        return len(self.phones_list)

    # def num_syllables(self):
    #   return len(self.syllables_list)

    def get_num_vowels(self, language):
        return num_vowel_phones(self.phones_list, language)

    def get_nth_vowel_phone(self, language, n=0):
        count = -1
        for i, ph in enumerate(self.phones_list):
            if is_vowel_sound(ph, language):
                count = count + 1
            if count == n:
                return ph, i
        else:
            return None

    # def get_syllable_of_nth_phone(self, n):
    #   count = -1
    #   for i, syl in enumerate(self.syllables_list):
    #     for ph in syl:
    #       count = count + 1
    #       if count == n:
    #         return i
    #   return None


# Save the phone cache to disk
def save_phone_cache(cache, filename):
    with open(filename, 'w') as f:
        for key in list(cache.keys()):
            festival, espeak = cache[key]
            f.write(key + '\t' + festival + '\t' + espeak + '\n')


# Load the phone cache from disk
def load_phone_cache(filename):
    cache = {}
    for line in open(filename, 'r'):
        line = line.strip()
        split_line = line.split('\t')
        if len(split_line) >= 3:
            word, festival, espeak = split_line
            cache[word] = (festival, espeak)
    return cache


# def load_near_rhyme_dictionary(filename):
#   rhyme_dict = {}
#   for line in tf.io.gfile.GFile(filename, 'r'):
#     line = line.split('\t')
#     key = line[0].strip()
#     val = line[1].strip()
#     if key not in rhyme_dict:
#       rhyme_dict[key] = []
#     rhyme_dict[key].append(val)
#   return rhyme_dict

# Return true if two words are perfect rhymes of each other
def perfect_rhyme(word1: Union[Dict, str, Phonetic], word2: Union[Dict, str, Phonetic]):
    if isinstance(word1, Phonetic):
        phonetic1 = word1
    elif isinstance(word1, dict):
        word1, e1, f1 = word1['w'], word1['e'], word1['f']
        phonetic1 = Phonetic(word1, f1, e1)
    else:
        e1, f1 = None, None
        phonetic1 = Phonetic(word1, f1, e1)
    if isinstance(word2, Phonetic):
        phonetic2 = word2
    elif isinstance(word2, dict):
        word2, e2, f2 = word2['w'], word2['e'], word2['f']
        phonetic2 = Phonetic(word2, f2, e2)
    else:
        e2, f2 = None, None
        phonetic2 = Phonetic(word2, f2, e2)

    stress1 = phonetic1.all_stresses[-1]
    stress2 = phonetic2.all_stresses[-1]
    if phonetic1.num_phones() - stress1 != phonetic2.num_phones() - stress2:
        return False
    for i in range(phonetic1.num_phones() - stress1):
        phone1 = phonetic1.phones_list[i + stress1]
        phone2 = phonetic2.phones_list[i + stress2]
        if phone1 != phone2:
            return False
    return True


# Let's pretend that these phones are the same
# (In addition all vowel phones that start with the same letter will be considered the same)
NEAR_SETS = [['er', 'r'],
             # ['ih', 'eh'],
             ['eh', 'ah'],
             ['er', 'ax'],
             ['ae', 'ey'],
             ['t', 'f'],
             ['b', 'k'],
             ['p', 'z', 'th'],
             ['s', 'st', 'sk'],
             ['nt', 'ns'],
             ['n', 'ng']
             ]

# These phonemes start with the same letter but should not be considered eqivalent
FAR_SETS = [['aw', 'ay', 'ax'],
            ['ax', 'ao']]


# Determine if phonemes are a near match
def near_match(phone1, phone2, language):
    # Safety check
    if (phone1 is None) or (phone2 is None) or (len(phone1) == 0 and len(phone2) > 0) or (
            len(phone2) == 0 and len(phone1) > 0):
        return False
    if phone1 == phone2:
        # Phones are the same
        return True
    elif is_vowel_sound(phone1, language) and is_vowel_sound(phone2, language) and phone1[0] == phone2[0]:
        # Vowel sounds start with the same letter
        # Check that they are not in the same far set
        for fs in FAR_SETS:
            if phone1 in fs and phone2 in fs:
                return False
        return True
    elif language == 'english':
        # Check to see if the pair of phones are in the same near_set
        for ns in NEAR_SETS:
            if phone1 in ns and phone2 in ns:
                return True
    return False


# Check if every phone in list 1 is a near match to corresponding phone in list 2
def near_matches(phone_list1, phone_list2):
    if len(phone_list1) != len(phone_list2):
        return False
    else:
        for i in range(len(phone_list1)):
            if not near_match(phone_list1[i], phone_list2[i]):
                return False
    return True


# If you turn this on, it will print information about the phoneme analysis being conducted
# every time near_rhyme() is called. Not recommended
VERBOSE = False


# Verbose print allows for debug printing to be turned off
def vprint(*args):
    if VERBOSE:
        print(' '.join([str(a) for a in args]))


# Determine if word1 and word2 are near rhymes
# if last_consonant is False, we won't check for near matches of the last consonant
def near_rhyme(word1, word2, language, last_consonant=True):
    # for chinese we use a different rhyming rule
    if perfect_rhyme(word1, word2):
        return True
    if isinstance(word1, Phonetic):
        phonetic1 = word1
    elif isinstance(word1, dict):
        word1, e1, f1 = word1['w'], word1['e'], word1['f']
        phonetic1 = Phonetic(word1, f1, e1)
    else:
        e1, f1 = None, None
        phonetic1 = Phonetic(word1, f1, e1)
    if isinstance(word2, Phonetic):
        phonetic2 = word2
    elif isinstance(word2, dict):
        word2, e2, f2 = word2['w'], word2['e'], word2['f']
        phonetic2 = Phonetic(word2, f2, e2)
    else:
        e2, f2 = None, None
        phonetic2 = Phonetic(word2, f2, e2)

    if len(phonetic1.phones_list) == 0 or len(phonetic2.phones_list) == 0:
        return False
    stress1 = phonetic1.all_stresses[-1]  # The last stressed phone
    stress2 = phonetic2.all_stresses[-1]  # The last stressed phone

    # Word = a v x w c
    # a = prefix (not important)
    # v = last stressed vowel
    # x = any number of phones between v and w
    # w = last vowel
    # c = consonant after last vowel

    # Last stressed vowel (v)
    v1_index = stress1
    v2_index = stress2

    v1 = phonetic1.phones_list[v1_index]
    v2 = phonetic2.phones_list[v2_index]
    vprint('v:', v1, v1_index, v2, v2_index)

    # Last vowel (w)
    w1, w1_index = phonetic1.get_nth_vowel_phone(language, phonetic1.get_num_vowels(language) - 1)
    w2, w2_index = phonetic2.get_nth_vowel_phone(language, phonetic2.get_num_vowels(language) - 1)
    vprint('w:', w1, w1_index, w2, w2_index)

    # Consonants after last vowel (c)
    c1_index = w1_index + 1
    c1 = ''.join(phonetic1.phones_list[c1_index:phonetic1.num_phones()])
    c2_index = w2_index + 1
    c2 = ''.join(phonetic2.phones_list[c2_index:phonetic2.num_phones()])
    vprint('c:', c1, c1_index, c2, c2_index)

    # phones between v and w (x)
    x1 = phonetic1.phones_list[v1_index + 1:w1_index]
    x2 = phonetic2.phones_list[v2_index + 1:w2_index]
    vprint('x:', x1, x2)

    # p = first phone in x
    # q = last phone in x
    p1 = None
    q1 = None
    p2 = None
    q2 = None
    if len(x1) > 0:
        p1 = x1[0]
        q1 = x1[-1]
    if len(x2) > 0:
        p2 = x2[0]
        q2 = x2[-1]
    vprint('p,q:', p1, q1, p2, q2)

    if not near_match(w1, w2, language):
        vprint('w fail')
        return False
    elif not near_match(v1, v2, language):
        vprint('v fail')
        return False
    elif False and len(c1) != len(c2) and (len(c1) > 1 or len(c2) > 1):
        vprint('c not same length - fail')
        return False
    elif last_consonant and not near_match(c1, c2, language):
        vprint('cs dont match - fail')
        return False
    elif len(x1) == 0 and len(x2) == 0:
        vprint("no x - match")
        return True
    elif len(x1) == 1 and len(x2) == 1 and near_match(x1[0], x2[0], language):
        vprint('single x - match')
        return True
    elif len(x1) > 0 and len(x2) > 0 and num_vowel_phones(x1, language) != num_vowel_phones(x2, language):
        vprint('num vowel phones in x - fail')
        return False
    elif near_match(p1, p2, language) and near_match(q1, q2, language):
        vprint('ps or qs - match')
        return True
    return False


def safe_near_rhyme(a, b):
    a = re.sub(r'[^a-zA-Z+ ]', '', a)
    b = re.sub(r'[^a-zA-Z+ ]', '', b)
    if len(a) < 1 or len(b) < 1:
        return False
    else:
        return near_rhyme(a, b)
