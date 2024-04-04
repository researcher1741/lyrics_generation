from typing import Any, List

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
import logging

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Model
# from src.lyrics_generation_utils.constants import LYRICS
from src.lyrics_generation_utils.constants import LYRICS
from src.lyrics_generation_utils.lr_schedulers import LinearWarmupCosineAnnealingLR
from src.lyrics_generation_utils.models_utils import get_lyrics_modelling_model


class PretrainingModule(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(conf)
        self.tokenizer = AutoTokenizer.from_pretrained(conf.model.tokenizer_path)
        self.model_name = conf.model.pretrained_model_name_or_path
        if not conf.model.from_pretrained:
            config = AutoConfig.from_pretrained(conf.model.pretrained_model_name_or_path)
            config.vocab_size = len(self.tokenizer)
            config.bos_token_id = self.tokenizer.bos_token_id
            config.eos_token_id = self.tokenizer.eos_token_id
            self.model = AutoModelForCausalLM.from_config(config)
        else:
            self.model = get_lyrics_modelling_model(self.model_name, self.tokenizer, conf.model.from_pretrained, conf)
        self.generated = 0
        self.lyrics_token_id = self.tokenizer.encode(LYRICS, add_special_tokens=False)[0]
        self.log_every_n_steps = conf.train.pl_trainer.log_every_n_steps if hasattr(conf.train.pl_trainer,
                                                                                    'log_every_n_steps') else 50
        self.num_steps = 0
        self.train_loss = 0

    def forward(self, batch) -> dict:
        # print(f"batch['input_ids'] {batch['input_ids'].size()}, {batch['input_ids']}")
        # print(f"batch.get('attention_mask', None) {batch.get('attention_mask', None).size()}, {batch.get('attention_mask', None)}")
        # print(f"batch.get('labels', None) {batch.get('labels', None).size()}, {batch.get('labels', None)}")
        output_dict = self.model(input_ids=batch['input_ids'],
                                 attention_mask=batch.get('attention_mask', None),
                                 labels=batch.get('labels', None))
        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # batch['labels'] = batch['input_ids']
        forward_output = self.forward(batch)
        self.log("loss", forward_output["loss"], sync_dist=True)
        self.train_loss += forward_output["loss"].item()
        self.num_steps += 1
        sch: LinearWarmupCosineAnnealingLR = self.lr_schedulers()
        sch.step()
        lr = sch.get_last_lr()[0]
        self.log('lr', lr, sync_dist=True, prog_bar=True)
        if self.trainer.progress_bar_callback is None and (
                self.num_steps % self.log_every_n_steps == 0 or self.num_steps < 10):
            print('=' * 40)
            print(f'Current Epoch: {self.current_epoch}')
            print(f'Num Steps: {self.num_steps}')
            print(f'Learning Rate: {lr}')
            print(f'Loss: {self.train_loss / self.num_steps:.2f}')
        return forward_output["loss"]

    def generate(self, in_data, **generation_args):
        logger = logging.getLogger('transformers.generation_utils')
        old_level = logger.level
        logger.setLevel(logging.ERROR)
        if generation_args is None or len(generation_args) == 0:
            from pathlib import Path
            import os
            import yaml
            path = open(os.path.join(Path(__file__).parents[3].__str__(), 'conf/generation/nucleus_gpt2.yaml'))
            generation_args = yaml.full_load(path)
        if in_data.shape[1] >= self.hparams.data.max_len:  # this seems to happen... bah
            return []
        generations = self.model.generate(inputs=in_data,
                                          **generation_args
                                          )
        logger.setLevel(old_level)

        return generations

    def validation_step(self, batch: dict, batch_idx: int):
        # batch['labels'] = batch['input_ids']

        forward_output = self.forward(batch)
        self.log("val_loss", forward_output["loss"], sync_dist=True)
        if self.generated <= 25:
            in_data = batch['input_ids']
            aux = list()
            # print('batch', batch.keys())
            # print('in_data', in_data)
            # print('self.lyrics_token_id', self.lyrics_token_id)
            max_len = torch.max(torch.where(in_data == self.lyrics_token_id)[1])
            for x in in_data:
                stop_idx = torch.where(x == self.lyrics_token_id)[0]
                aux.append(torch.cat(
                    [torch.zeros(max_len - stop_idx, device=in_data.device).long() + self.tokenizer.pad_token_id,
                     x[:stop_idx]], 0))
            in_data = torch.stack(aux, 0)
            generations = self.generate(in_data)
            self.generated += in_data.shape[0]
            return forward_output["loss"], in_data.tolist(), generations
        return forward_output["loss"], [], []

    def validation_epoch_end(self, all_generations: List[Any]) -> None:
        self.generated = 0
        _, in_data, all_generations = zip(*all_generations)
        in_data = [self.tokenizer.decode(in_d) for data in in_data for in_d in data]
        all_generations = [self.tokenizer.decode(g) for gens in all_generations for g in gens]
        logger = logging.getLogger('transformers.generation_utils')
        old_level = logger.level
        logger.setLevel(logging.INFO)
        columns = ['Prompt', 'Generation']
        data = []
        for id, gen in zip(in_data, all_generations):
            logger.info('Prompt:')
            logger.info(id.replace('</s>', '').replace('<pad>', ''))
            logger.info('Generation')
            logger.info(gen.replace(id, '').replace('</s>', '').replace('<pad>', ''))
            data.append([id, gen.replace(id, '').replace('</s>', '').replace('<pad>', '')])
            logger.info('=' * 40)
        logger.setLevel(old_level)
        self.logger.log_text(key='generations', columns=columns, data=data)

    def test_step(self, batch: dict, batch_idx: int):
        # batch['labels'] = batch['input_ids']

        forward_output = self.forward(batch)
        self.log("test_loss", forward_output["loss"], sync_dist=True)
        ## FIXME, the loss that we get is the average within the batch, ideally, we should compute
        # the loss for each song (and thus the perplexity) and then average by the number of songs.
        # this is not possible in transformers, we need to recompute the loss from logits.
        ## Furthermore, I should evaluate only the interesting part, i.e., the one after the prompt.
        return torch.exp(forward_output['loss'])

    def test_epoch_end(self, song_perplexities: List[Any]):
        song_perplexities = torch.stack(song_perplexities)  #
        avg_song_perplexity = song_perplexities.sum() / song_perplexities.shape[0]
        self.log('average_song_perplexity', avg_song_perplexity.item(), sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.hparams.train.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, 50_000, self.hparams.train.pl_trainer.max_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler':
                {
                    'scheduler': scheduler
                }
        }
