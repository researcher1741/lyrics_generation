from typing import Any, List

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
import logging
from torch.nn import CrossEntropyLoss
from transformers import LogitsProcessorList
# from src.lyrics_generation_utils.constants import SENTENCE_END
from src.lyrics_generation_utils.constants import SENTENCE_END
from src.lyrics_generation_utils.lr_schedulers import LinearWarmupCosineAnnealingLR
from src.lyrics_generation_utils.utils import SchemaEnforcingLogitsProcessor, get_lyrics_tokenizer
from src.lyrics_generation_utils.models_utils import get_lyrics_modelling_model


class MultitaskLyricsModule(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(conf)
        self.tokenizer = get_lyrics_tokenizer(conf)

        # backward compatibility
        self.model_name = conf.model.pretrained_model_name_or_path if "pretrained_model_name_or_path" \
                                                                      in conf.model \
            else conf.model.pretrained_model_name
        from_pretrained = conf.model.from_pretrained if "from_pretrained" in conf.model else True
        self.model = get_lyrics_modelling_model(self.model_name, self.tokenizer, from_pretrained)
        self.generated = 0
        self.sentence_end_id = self.tokenizer.encode(SENTENCE_END, add_special_tokens=False)[0]

    def forward(self, batch) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        output_dict = self.model(
            input_ids=batch['input_ids'],
            decoder_input_ids=batch.get('decoder_input_ids'),
            tasks_labels=batch.get('task_labels'),
            labels=batch.get('labels')
        )
        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # if batch_idx < 3:
        #     return None
        forward_output = self.forward(batch)
        self.log("loss", forward_output["loss"], sync_dist=True)
        self.__log_losses(forward_output, 'train')
        sch: LinearWarmupCosineAnnealingLR = self.lr_schedulers()
        sch.step()
        lr = sch.get_last_lr()[0]
        self.log('lr', lr, sync_dist=True, prog_bar=True)
        return forward_output["loss"]

    def generate(self, in_data, schema=None, **generation_args):
        logger = logging.getLogger('transformers.generation_utils')
        old_level = logger.level
        logger.setLevel(logging.ERROR)
        if generation_args is None or len(generation_args) == 0:
            from pathlib import Path
            import os
            import yaml
            path = open(os.path.join(Path(__file__).parents[3].__str__(), 'conf/generation/nucleus_gpt2.yaml'))
            generation_args = yaml.full_load(path)
        if schema is not None:
            logits_processors = LogitsProcessorList()
            num_beams = generation_args['num_beams']
            aux = []
            for s in schema:
                aux.extend([s] * num_beams)
            logits_processors.append(SchemaEnforcingLogitsProcessor(aux, self.sentence_end_id, self.tokenizer))
        else:
            logits_processors = None
        generations = self.model.bart.generate(inputs=in_data,
                                               # logits_processor=logits_processors,
                                               **generation_args
                                               )
        logger.setLevel(old_level)
        return generations

    def __log_losses(self, forward_output, split):
        losses = [(k, v) for k, v in forward_output.items() if '_loss' in k]
        for k, v in losses:
            task_name = k.replace('_loss', '')
            self.log(f'{task_name}_{split}_loss', v, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int):
        forward_output = self.forward(batch)
        self.__log_losses(forward_output, 'val')
        if self.generated <= 25:
            if 'bart-' in self.model_name:
                in_data = batch['input_ids']
            else:
                in_data = batch['input_ids'][:, :20]
            ## TODO get schema and pass it to generate
            generations = self.generate(in_data, schema=batch['schema'])
            self.generated += in_data.shape[0]
            return forward_output["loss"], in_data, generations
        self.log('val_loss', forward_output['loss'])
        return forward_output["loss"], [], [], [], []

    def validation_epoch_end(self, all_generations: List[Any]) -> None:
        self.generated = 0
        _, in_data, all_generations = zip(*all_generations)
        in_data = [self.tokenizer.decode(in_d[in_d != self.tokenizer.pad_token_id]) for data in in_data for in_d in
                   data]
        all_generations = [self.tokenizer.decode(g[g != self.tokenizer.pad_token_id]) for gens in all_generations for g
                           in gens]
        logger = logging.getLogger('transformers.generation_utils')
        old_level = logger.level
        logger.setLevel(logging.INFO)
        columns = ['Prompt', 'Generation']
        data = []
        for id, gen in zip(in_data, all_generations):
            logger.info('Prompt:')
            logger.info(id)
            logger.info('Generation')
            logger.info(gen.replace(id, ''))
            data.append([id, gen.replace(id, '')])
            logger.info('=' * 40)
        logger.setLevel(old_level)
        self.logger.log_text(key='generations', columns=columns, data=data)

    def test_step(self, batch: dict, batch_idx: int):
        forward_output = self.forward(batch)

        self.__log_losses(forward_output, 'test')
        logits = forward_output.logits
        labels = batch['labels']
        cel = CrossEntropyLoss(reduction='none', ignore_index=-100)
        losses = cel(logits.view(-1, logits.shape[-1]), labels.view(-1)).view(labels.shape[0], -1)
        song_losses = losses.sum(-1) / (labels != -100).sum(-1)
        # self.log('test_loss', forward_output['loss'], prog_bar=True)
        return torch.exp(song_losses)

    def test_epoch_end(self, song_perplexities: List[Any]):
        song_perplexities = torch.cat(song_perplexities)  #
        avg_song_perplexity = song_perplexities.mean()
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
