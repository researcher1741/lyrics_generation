import hydra
import omegaconf

from src.ssl import no_ssl_verification
from train_encoder_decoder import train
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
os.environ['MASTER_PORT'] = '9999'
os.environ['CURL_CA_BUNDLE'] = ''


@hydra.main(config_path="../../../conf", config_name="root_pretrain_t5_large")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == '__main__':
    import requests
    # Monkey patch the requests functions
    from functools import partial

    # Monkey patch the requests functions
    requests.request = partial(requests.request, verify=False)
    requests.get = partial(requests.get, verify=False)
    requests.head = partial(requests.head, verify=False)
    requests.post = partial(requests.post, verify=False)
    requests.put = partial(requests.put, verify=False)
    requests.patch = partial(requests.patch, verify=False)
    requests.delete = partial(requests.delete, verify=False)
    # Remove warning
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
    main()
