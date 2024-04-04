import omegaconf
import hydra
import os
os.environ['TOKENIZERS_PARALLELISM']='true'
os.environ['REQUESTS_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'
os.environ['MASTER_PORT'] = '9999'
os.environ['CURL_CA_BUNDLE'] = ''

from src.lyrics_generation.pretrainers.pretrain_gpt2_zh import train


## TODO when loading the config for GPT2 reassign the vocab_size to match the one of the vocabulary.

@hydra.main(config_path="../../../conf", config_name="root_pretrain_gpt2_zh2")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
