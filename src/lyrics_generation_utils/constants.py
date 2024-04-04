GENRE = '<genre>'
ARTIST = '<artist>'
TITLE = '<title>'
EMOTIONS = '<emotions>'
TOPICS = '<topics>'

LYRICS = '<lyrics>'
LANG = '<lang>'
END_LYRICS = '</lyrics>'
CHORUS = '<chorus>'
BLOCK_END = '<block_end>'
SENTENCE_END = '<sentence_end>'
TAG_END = '<tag_end>'
CONTINUE_TOKEN = '<continue>'
GENERATE = '<generate>'
NUM_SYLLABLES = '<num_syllables>'
SCHEMA = '<schema>'
SEP = '<SEP>'
RHYME_TOKENS = [f'RHYME_{chr(ord("A") + i)}' for i in range(26)]
LYRICS_SPECIAL_TOKENS = [GENRE, ARTIST, TITLE, EMOTIONS, TOPICS, LYRICS, LANG, END_LYRICS, CHORUS, BLOCK_END,
                         SENTENCE_END, CONTINUE_TOKEN, GENERATE, NUM_SYLLABLES, TAG_END, SCHEMA, SEP] + RHYME_TOKENS

SEMI_AUTOMATIC_CLUSTERING = \
    {'romantic': {'dreamy', 'romantic'},
     'bittersweet': {'bittersweet'},
     'calm': {'calm', 'quiet'},
     'sad': {'angry', 'sad'},
     'energetic': {'energetic'},
     'mellow': {'mellow', 'smooth', 'soft'},
     'melancholic': {'melancholic', 'melancholy'},
     'relax': {'relax', 'relaxing', 'soothing'},
     'happy': {'happy', 'fun', 'funny', 'party', 'sexy'},
     'aggressive': {'aggressive', 'dark', 'intense'}}

SEMI_AUTOMATIC_REVERSE_CLUSTERING = dict()
for k, v in SEMI_AUTOMATIC_CLUSTERING.items():
    for vv in v:
        if vv in SEMI_AUTOMATIC_REVERSE_CLUSTERING:
            raise RuntimeError("Error in clustering")
        SEMI_AUTOMATIC_REVERSE_CLUSTERING[vv] = k
