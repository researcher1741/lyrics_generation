from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import omegaconf
import hydra
from transformers import AutoConfig
from src.lyrics_generation.decoder_module import DecoderModule
from src.lyrics_generation.encoder_decoder_module import EncoderDecoderModule
from src.lyrics_generation.multitask_lyrics_module import MultitaskLyricsModule
from src.lyrics_generation_utils.init_utils import init_everything_for_evaluation
import torch

from src.lyrics_generation_utils.utils import get_lyrics_tokenizer
with initialize(config_path="../../conf", ):
    conf = compose(config_name='root_t5', overrides=["+out_dir=/tmp",
                "+checkpoint_path='/lyrics_generation/experiments/genius_section-v0.3/t5v1.1-large-data-v0.3/checkpoints/epoch=11-step=399087.ckpt'",
                "+model.tokenizer_path='/lyrics_generation/experiments/genius_section-v0.3/t5v1.1-large-data-v0.3/tokenizer/'",
                "train.pl_trainer.gpus=[1]",
                "model.from_checkpoint=True"])

current_conf_path='/tmp/hparams.yaml'
with open(current_conf_path, 'w') as writer:
    writer.write(OmegaConf.to_yaml(conf))
tokenizer = get_lyrics_tokenizer(conf)
hf_conf = AutoConfig.from_pretrained(conf.model.pretrained_model_name_or_path)
model_cls = None
if hf_conf.is_encoder_decoder:
    if 'multitask' in conf.model.pretrained_model_name_or_path:
        model_cls = MultitaskLyricsModule
    else:
        model_cls = EncoderDecoderModule
else:
    model_cls = DecoderModule
pl_module = model_cls.load_from_checkpoint(conf.checkpoint_path, hparams_file=current_conf_path, strict=True)
gen_args = dict(
        max_length=conf.generation.max_length if 'max_length' in conf.generation else 50,
        num_beams=conf.generation.num_beams if 'num_beams' in conf.generation else 5,
        no_repeat_ngram_size=conf.generation.no_repeat_ngram_size if 'generation_no_repeat_ngram_size' in conf.generation else 2,
        early_stopping=conf.generation.no_repeat_ngram_size if 'generation_no_repeat_ngram_size' in conf.generation else True,
        decoder_start_token_id=tokenizer.pad_token_id,
        do_sample=False,
        forced_bos_token_id=None
)

input_str = '<title> Hook it Up<artist> R. Kelly<genre> R&B<topics> ass<schema>RHYME_A RHYME_B RHYME_C RHYME_C RHYME_D RHYME_E RHYME_F RHYME_G RHYME_H</s>'
input_ids = tokenizer.encode(input_str, add_special_tokens=False)
input_ids = torch.LongTensor(input_ids)
schema = [x.replace('RHYME_', '') for x in input_str.split('<schema>')[1].split()]
generation = pl_module.generate(input_ids.unsqueeze(0), schema = [schema], batch=None, **gen_args)
pl_module.model.generate()
def decode(generation):
    str_gen = tokenizer.batch_decode(generation)[0]
    str_gen = str_gen.replace('<sentence_end>', '<sentence_end>\n')
    print(str_gen)

decode(generation)