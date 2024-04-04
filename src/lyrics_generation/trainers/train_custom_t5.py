import hydra
import omegaconf
from train_encoder_decoder import train
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
os.environ['MASTER_PORT'] = '9999'
os.environ['CURL_CA_BUNDLE'] = ''


@hydra.main(config_path="../../../conf", config_name="root_custom_t5")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == '__main__':
    main()
