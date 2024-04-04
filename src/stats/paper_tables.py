from tqdm import tqdm
import jsonlines
import os

def compute_stats(split, folder):
    with_topics = 0
    with_emotions = 0
    with_genres = 0
    num_examples = 0
    lyrics_tokens_average = 0
    sentence_token_len_average = 0
    num_sentences_average = 0
    tot_num_sentences = 0
    languages = set()
    with jsonlines.open(os.path.join(folder, split + '.jsonl')) as lines:
        for line in tqdm(lines, desc='computing stats'):
            num_examples += 1
            verses = [verse for verse in line['lyrics'] if len(verse) > 1 or not verse[0].endswith('<tag_end>')]
            topics = line['topics']
            genre = line['genre']
            emotions = line['emotions']
            if 'lang' in line:
                languages.add(line['lang'])

            if len(topics) > 0:
                with_topics += 1
            if len(emotions) > 0:
                with_emotions += 1
            if len(genre) > 0:
                with_genres += 1
            num_sentences_average += len(verses)
            for verse in verses:
                lyrics_tokens_average += len(verse)
                sentence_token_len_average += len(verse)
                tot_num_sentences += 1
    sentence_token_len_average = sentence_token_len_average / tot_num_sentences
    lyrics_tokens_average = lyrics_tokens_average / num_examples
    num_sentences_average =  num_sentences_average / num_examples
    return {
        'num_examples': num_examples,
        'with_genre': f'{with_genres} / {with_genres / num_examples * 100:.2f}%',
        'with_emotions': f'{with_emotions} / {with_emotions / num_examples * 100:.2f}%',
        'with_topics': f'{with_topics} / {with_topics / num_examples * 100:.2f}%', 
        'lyrics_token_average': lyrics_tokens_average, 
        'sentence_token_average': sentence_token_len_average, 
        'num_sentences_average': num_sentences_average,
        'num_languages': len(languages)
    }
    

def dataset_stats(folder):
    for split in ['train', 'dev', 'test']:
        stats = compute_stats(split, folder)
        print('split:', split)
        for k, v in stats.items():
            print(f'\t{k}: {v}')
        print('=' * 40)


if __name__ == '__main__':
    #dataset_stats('./LG/DATA/genius_section_0.2')
    dataset_stats('data/multilingual_section_dataset/')
        
        
