from phonemizer.backend import BACKENDS

from src.lyrics_datasets.multilingual_processing.blockify_dataset import build_rhyming_schema


if __name__ == '__main__':
    phonemizer = BACKENDS['espeak'](
            'en-us',
            with_stress=True
            )
    sentences = '''
So, so you think you can tell\n
Heaven from hell?\n
Blue skies from pain?\n
Can you tell a green field\n
From a cold steel rail?\n
A smile from a veil?\n
Do you think you can tell?\n
'''.strip().split('\n')
    schema, new_sentences = build_rhyming_schema(sentences, 'english', phonemizer)
    print(schema)
