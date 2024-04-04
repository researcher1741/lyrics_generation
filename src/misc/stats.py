from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import omegaconf
from src.lyrics_generation.multitask_lyrics_data_module import MultitaskLyricsDataModule

from src.lyrics_generation_utils.init_utils import init_everything_for_evaluation
from src.lyrics_generation_utils.utils import get_lyrics_tokenizer
def print_genius_stats():
    with initialize(config_path="../../conf", ):
        conf = compose(config_name='root_multitask')
        tokenizer = get_lyrics_tokenizer(conf)
        pl_data_module = MultitaskLyricsDataModule(tokenizer, conf)
        pl_data_module.prepare_data()
        for dataset in [pl_data_module.training_dataset, pl_data_module.dev_dataset, pl_data_module.test_dataset]:
            genres = [x for x in dataset.examples['encoded_genre'] if len(x) > 0]
            emotions = [x for x in dataset.examples['encoded_emotion_tags'] if len(x[0]) > 0]
            topics = [x for x in dataset.examples['encoded_topics'] if len(x[0]) > 0]
            num_examples = len(dataset)
            print(dataset.split_name)
            print(f'examples {num_examples}')
            print(f'genres {len(genres)} {len(genres)/num_examples * 100:.2f}')
            print(f'emotions {len(emotions)} {len(emotions)/num_examples * 100:.2f}')
            print(f'topics {len(topics)} {len(topics)/num_examples * 100:.2f}')
            print('='* 40)



if __name__ == '__main__':
    print_genius_stats()
        
    