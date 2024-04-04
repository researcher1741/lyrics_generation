import hydra
import omegaconf
from src.lyrics_generation.trainers.train_encoder_decoder import train

@hydra.main(config_path="../../../conf", config_name="root_pretrain_t5")
def main(conf: omegaconf.DictConfig):
    train(conf)

if __name__ == '__main__':
    main()
