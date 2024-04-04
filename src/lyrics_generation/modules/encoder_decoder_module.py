from typing import Any, List
from torch.distributions import Categorical
from collections import Counter, defaultdict
import pytorch_lightning as pl
import torch
from transformers import AutoConfig
from torch.optim import AdamW
import logging
from transformers import LogitsProcessorList
# from src.lyrics_generation_utils.constants import RHYME_TOKENS, SENTENCE_END, SEP
from src.lyrics_generation_utils.constants import RHYME_TOKENS, SENTENCE_END, SEP
from src.lyrics_generation_utils.models_utils import get_lyrics_modelling_model
import math
from src.lyrics_generation_utils.utils import FirstWordSEPLogitProcessor, SchemaEnforcingLogitsProcessor, \
    get_info_logger, get_lyrics_tokenizer
import re
import math
from pytorch_lightning.loggers import WandbLogger

import pronouncing
import inspect
import numpy as np

from src.misc.lr_scheduler import LinearLR


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__()
        print(conf)
        self.save_hyperparameters(conf)
        self.tokenizer = get_lyrics_tokenizer(conf)
        self.alpha = conf.train.alpha if hasattr(conf.train, 'alpha') else 1.0
        self.decay_alpha = conf.train.decay_alpha if hasattr(conf.train, 'decay_alpha') else False
        self.alpha_target = conf.train.alpha_target if hasattr(conf.train, 'alpha_target') else None
        self.decoder_start_token_id = AutoConfig.from_pretrained(
            self.hparams.model.pretrained_model_name_or_path).decoder_start_token_id

        self.decay_alpha_num_steps = conf.decay_alpha_num_steps if hasattr(conf.train,
                                                                           'decay_alpha_num_steps') else None
        self.alpha_decay_rate = None
        self.console_logger = get_info_logger(__name__)
        if self.decay_alpha and self.decay_alpha_num_steps is not None and self.alpha_target is not None:
            if hasattr(self.hparams.train, 'use_reinforce') and self.hparams.train.use_reinforce:
                self.alpha_decay_rate = math.abs(self.alpha - self.target_alpha) / self.decay_alpha_num_steps
                self.reward_matrix = self._compute_reward_matrix()

        # backward compatibility
        self.special_sep_id = self.tokenizer.encode(SEP, add_special_tokens=False)[0]
        self.model_name = model_name = conf.model.pretrained_model_name_or_path if "pretrained_model_name_or_path" in conf.model else conf.model.pretrained_model_name
        from_pretrained = conf.model.from_pretrained if "from_pretrained" in conf.model else True
        self.model = get_lyrics_modelling_model(model_name, self.tokenizer, from_pretrained, conf=conf)
        self.sentence_end_id = self.tokenizer.encode(SENTENCE_END, add_special_tokens=False)[0]
        self.generated = 0
        self.rhyming_letter2tokid = dict()
        self.log_every_n_steps = conf.train.pl_trainer.log_every_n_steps if hasattr(conf.train.pl_trainer,
                                                                                    'log_every_n_steps') else 50
        self.num_steps = 0
        self.train_losses = Counter()
        self.force_schema = conf.model.force_schema
        self.force_first_word_and_sep = conf.data.version == '0.3' or conf.data.version == '0.2.2' or conf.data.version == '0.2.1'
        for i, rhyming_token in enumerate(RHYME_TOKENS):
            tokid = self.tokenizer.encode(rhyming_token, add_special_tokens=False)[0]
            self.rhyming_letter2tokid[chr(ord('A') + i)] = tokid
        self.tasks = conf.train.tasks
        self.task_probabilities = conf.train.task_probs
        self.training_random_choices = self.__init_random_choices()

    def _compute_reward_matrix(self):
        vocab = self.tokenizer.vocab
        words = list(np.empty(len(vocab)))
        reward_matrix = np.zeros((len(vocab), len(vocab))) + -1
        for k, v in vocab.items():
            words[k] = v
        for i in range(len(words)):
            w1 = words[i]
            for j in range(i + 1, len(words)):
                w2 = words[j]
                if self.__do_rhyme(w1, w2):
                    reward_matrix[i, j] = 1, 0
        return reward_matrix

    def __init_random_choices(self):
        return list(np.random.choice(self.tasks, 100000, replace=True, p=self.task_probabilities))

    def forward(self, batch) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        arg_names, *_ = inspect.getfullargspec(self.model.forward)
        custom_batch = {k: batch[k] for k in arg_names if k in batch}
        output_dict = self.model(**custom_batch)
        return output_dict

    def training_step(self, all_tasks_batch: dict, batch_idx: int) -> torch.Tensor:
        if len(self.training_random_choices) == 0:
            self.training_random_choices = self.__init_random_choices()
        task = self.training_random_choices.pop(0)
        if 'lyrics_generation' not in all_tasks_batch:
            batch = all_tasks_batch
        else:
            batch = all_tasks_batch[task]
        forward_output = self.forward(batch)
        if hasattr(self.hparams.train, 'use_reinforce') and \
                self.hparams.train.use_reinforce and \
                (not hasattr(self.hparams.train,
                             'no_reinforce_updates') or self.trainer.global_step >= self.hparams.train.no_reinforce_updates) and \
                batch['task_name'] == 'lyrics generation' and \
                self.alpha < 1:
            reinforce_loss, rewards = self.compute_reinforce_loss(batch, forward_output,
                                                                  self.hparams.train.reinforce_loss_samples)
            loss = self.alpha * forward_output['loss'] + (1 - self.alpha) * reinforce_loss
            reward = torch.mean(rewards)
            self.log('reinforce_loss', reinforce_loss, sync_dist=True, prog_bar=True)
            self.log('reinforce_rewards', reward, sync_dist=True)
            self.log("cross_entropy_loss", forward_output["loss"], sync_dist=True, prog_bar=True)
            if self.decay_alpha and self.decay_alpha_num_steps is not None and self.alpha_target is not None and self.global_step % self.trainer.accumulate_grad_batches == 0:
                # if reinforcement learning is enabled and alpha and alpha decay params are all set, then decay alpha value for each training step
                self.alpha = self.alpha - self.alpha_decay_rate
        else:
            loss = forward_output['loss']
            reinforce_loss = -1
            reward = -1
        self.log('loss', loss, sync_dist=True, prog_bar=True)
        if 'task_name' in batch:
            self.log(batch['task_name'] + '_loss', loss, sync_dist=True, prog_bar=True)
            self.train_losses[batch['task_name'] + '_loss'] += forward_output['loss'].item()
        self.train_losses['cross_entropy_loss'] += forward_output["loss"].item()

        if self.trainer.progress_bar_callback is None and (batch_idx % self.log_every_n_steps == 0 or batch_idx < 10):
            sch = self.lr_schedulers()
            lr = sch.get_last_lr()[0]
            self.console_logger.info('=' * 40)
            self.console_logger.info(f'Current Epoch: {self.current_epoch}')
            self.console_logger.info(f'Num Steps: {batch_idx}')
            self.console_logger.info(f'Learning Rate: {lr}')
            for k, v in self.train_losses.items():
                self.console_logger.info(f'{k}: {v / max(batch_idx, 1):.2f}')
        return loss

    def generate(self, in_data, schema, batch=None, **generation_args):
        logger = logging.getLogger('transformers.generation_utils')
        old_level = logger.level
        logger.setLevel(logging.ERROR)
        if generation_args is None or len(generation_args) == 0:
            conf = self.hparams
            generation_args = dict(
                max_length=conf.generation.max_length if 'max_length' in conf.generation else 300,
                output_scores=conf.generation.output_scores if 'output_scores' in conf.generation else True,
                num_beams=conf.generation.num_beams if 'num_beams' in conf.generation else 5,
                top_p=conf.generation.top_p if 'top_p' in conf.generation else 0.85,
                no_repeat_ngram_size=conf.generation.no_repeat_ngram_size if 'generation_no_repeat_ngram_size' in conf.generation else 2,
                early_stopping=conf.generation.no_repeat_ngram_size if 'generation_no_repeat_ngram_size' in conf.generation else True,
                decoder_start_token_id=self.decoder_start_token_id,
                do_sample=conf.generation.do_sample if 'do_sample' in conf.generation else True,
                forced_bos_token_id=None
            )
        logits_processors = LogitsProcessorList()
        if schema is not None and self.force_schema:
            num_beams = generation_args['num_beams']
            num_sentences = generation_args.get('num_return_sequences', 5)
            aux = []
            for s in schema:
                expand_dims = num_beams if not generation_args['do_sample'] else num_sentences
                aux.extend([s] * expand_dims)
            logits_processors.append(SchemaEnforcingLogitsProcessor(aux, self.sentence_end_id, self.tokenizer,
                                                                    trigger_after_schema_token=True,
                                                                    sample=generation_args.get('do_sample', False)))
        if self.force_first_word_and_sep:
            logits_processors.append(
                FirstWordSEPLogitProcessor(self.sentence_end_id, self.special_sep_id, self.tokenizer,
                                           sample=generation_args.get('do_sample', False)))
        custom_batch = dict()
        if batch is not None and self.hparams.model.use_custom:
            custom_batch = {k: batch[k] for k in ['decoder_rhyming_ids', 'decoder_position_ids'] if k in batch}
            generation_args['max_length'] = min(generation_args['max_length'],
                                                custom_batch['decoder_position_ids'].shape[1])
        generations = self.model.generate(inputs=in_data,
                                          logits_processor=logits_processors,
                                          pad_token_id=self.tokenizer.pad_token_id,
                                          **custom_batch,
                                          **generation_args
                                          )
        logger.setLevel(old_level)

        return generations

    def validation_step(self, all_tasks_batch: dict, batch_idx: int):
        input_data = defaultdict(list)
        all_generations = defaultdict(list)
        losses = {}
        in_data = None
        if 'lyrics_generation' not in all_tasks_batch:
            all_tasks_batch = {'lyrics_generation': all_tasks_batch}
        for task, batch in all_tasks_batch.items():
            forward_output = self.forward(batch)
            if task == 'lyrics_generation':
                key = 'val_loss'
            else:
                key = task + '_val_loss'
            self.log(key, forward_output["loss"], sync_dist=True)
            if hasattr(self.hparams.train,
                       'use_reinforce') and self.hparams.train.use_reinforce and task == 'lyrics_generation':
                reinforce_loss, reinforce_rewards = self.compute_reinforce_loss(batch, forward_output,
                                                                                self.hparams.train.reinforce_loss_samples)
                self.log("val_reinforce_loss", reinforce_loss, sync_dist=True)
                self.log('val_reinforce_rewards', torch.mean(reinforce_rewards), sync_dist=True)
            if self.generated < 25:
                if self.model.config.is_encoder_decoder:
                    in_data = batch['input_ids']
                else:
                    in_data = batch['input_ids'][:, :20]
                generations = self.generate(in_data, batch['schema'], batch)
                all_generations[task].extend(generations)
                input_data[task].extend(in_data)
            losses[task] = forward_output['loss']
        if self.generated < 25 and in_data is not None:
            self.generated += in_data.shape[0]
        return losses, input_data, all_generations

    def validation_epoch_end(self, losses_in_data_all_generations: List[Any]) -> None:
        self.generated = 0
        in_data_dict = defaultdict(list)
        generations_dict = defaultdict(list)
        for _, in_data_d, generations_d in losses_in_data_all_generations:
            for k in in_data_d.keys():
                in_data_dict[k].extend(in_data_d[k])
                generations_dict[k].extend(generations_d[k])

        for task in in_data_dict.keys():
            in_data = in_data_dict[task]
            all_generations = generations_dict[task]
            in_data = self.tokenizer.batch_decode(in_data)
            all_generations = self.tokenizer.batch_decode(all_generations)
            logger = logging.getLogger('transformers.generation_utils')
            old_level = logger.level
            logger.setLevel(logging.INFO)
            columns = ['Prompt', 'Generation']
            data = []
            logger.info('=' * 10 + ' ' + task.upper() + ' ' + '=' * 10)
            for id, gen in zip(in_data, all_generations):
                logger.info('Prompt:')
                logger.info(id.replace('</s>', '').replace('<pad>', ''))
                logger.info('Generation')
                logger.info(gen.replace(id, '').replace('</s>', '').replace('<pad>', ''))
                data.append([id, gen.replace(id, '').replace('</s>', '').replace('<pad>', '')])
                logger.info('=' * 40)
            logger.setLevel(old_level)
            if isinstance(self.logger, WandbLogger):
                self.logger.log_text(key=task + '_generations', columns=columns, data=data)

    def test_step(self, all_tasks_batch: dict, batch_idx: int):
        losses = {}
        reinforce_rewards = None
        for task, batch in all_tasks_batch.items():
            forward_output = self.forward(batch)
            self.log(task + "_test_loss", forward_output["loss"], sync_dist=True)
            if hasattr(self.hparams.train, 'reinforce_loss_samples') and task == 'lyrics_generation':
                reinforce_loss, reinforce_rewards = self.compute_reinforce_loss(batch, forward_output,
                                                                                self.hparams.train.reinforce_loss_samples)
                self.log('test_reinforce_rewards', torch.mean(reinforce_rewards), sync_dist=True)
            losses[task] = forward_output['loss'] if task != 'lyrics_generation' else torch.exp(forward_output['loss'])
        return losses, reinforce_rewards

    def test_epoch_end(self, losses_and_reinforce_rewards: List[Any]):
        all_losses, rewards = list(zip(*losses_and_reinforce_rewards))
        # if rewards is not None:
        #    rewards = torch.cat(rewards)
        #    rewards = torch.mean(rewards)

        song_perplexities = [l['lyrics_generation'] for l in all_losses]
        song_perplexities = torch.stack(song_perplexities)  #
        avg_song_perplexity = song_perplexities.sum() / song_perplexities.shape[0]
        task2loss = defaultdict(list)
        self.log('average_song_perplexity', avg_song_perplexity, sync_dist=True)
        # if rewards is not None:
        #    self.log('pair_average_rewards', rewards.item(), sync_dist=True)

        for task_dict in all_losses:
            for task, loss in task_dict.items():
                task2loss[task].append(loss.item())
        for task, losses in task2loss.items():
            self.log(f'average_{task}_loss', np.average(losses), sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.hparams.train.lr)
        # if int(pl.__version__.split('.')[1]) < 6:
        tot_steps = self.hparams.train.pl_trainer.max_steps // self.hparams.train.pl_trainer.accumulate_grad_batches
        # else:
        # tot_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil(tot_steps * 0.10)  # 10% of steps are for warming up
        scheduler = LinearLR(optimizer, total_iters=warmup_steps)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, 50_000, self.hparams.train.pl_trainer.max_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler':
                {
                    'scheduler': scheduler,
                    'interval': 'step'
                }
        }

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int):
        """Override this method to change the default behaviour of ``optimizer.zero_grad()``.

        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.

        Examples::

            # DEFAULT
            def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
                optimizer.zero_grad()

            # Set gradients to `None` instead of zero to improve performance.
            def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
                optimizer.zero_grad(set_to_none=True)

        See :meth:`torch.optim.Optimizer.zero_grad` for the explanation of the above example.
        """
        ## We need this ow T5 goes to segmentation fault. WHy? God knows. Anyway lightning suggest this
        ## to improve performance.. so we are happy.
        optimizer.zero_grad(set_to_none=True)

    def compute_reinforce_loss(self, batch, out, num_samples=5):
        logits = out['logits']  # logits are aligned with decoder_input_ids which are <s> + labels[:-1]
        schema = batch['schema']
        labels = [[l.item() for l in labels if l >= 0] for labels in batch['labels']]
        rhyming_indices = self.__get_rhyming_indices(schema)

        ### XXX THIS PART SHOULD BE DONE AT DATA PROCESSING TIME! NOT HERE. IT WAS BUT FOR SOME REASON IT SEEMS LIKE IT IS WRONG.
        pattern = re.compile(r'[;:\'"\]\}\{\[/\?\.>,<`~1!2@3#4$5%67890-=+_—\*\)\(&^%\$]+')
        sentence_end_idx = self.tokenizer.encode('<sentence_end>', add_special_tokens=False)[0]
        last_token_indices = [np.argwhere(np.array(l) == sentence_end_idx).squeeze() - 1 for l in labels]
        for elem_labels, arr in zip(labels, last_token_indices):
            for i in range(len(arr)):
                while pattern.match(self.tokenizer.decode(elem_labels[arr[i]]).strip()) is not None and elem_labels[
                    arr[i]] != self.sentence_end_id and arr[i] > 0:
                    arr[i] -= 1

        # # last_token_indices contains the indices, for each element in the batch and for each sentence in the
        # element of the last token before <sentence_end>. If the sentence # ends with punctuation that is discarded
        # untill a token of a word is found. N.B. that a token != word, so we match wether two tokens rhyme... which
        # might not be completely # correct as a word might be split in arbitrary portions and thus the phonemes
        # extraction and the identification of the rhyming phonemes (done with pronouncing) might be wrong. # we can
        # make up to this problem by merging the tokens going backwards untill a space is met. The same,
        # however cannot be done for the predicted words, indeed we need to generate # more tokens from the logits,
        # which does not really make sense, because of teacher forcing. We can mitigate this problem, by computing
        # this loss only for words that correspond to # a single token. OW we have to retrain a tokenizer and a model
        # based on a word-level tokenization (OOV words becomes then an issue).
        predictions_info = list()
        ## TODO MATTHIEU suggests to compute the reward for each word in the vocabulary rather than samplying
        ## this can be easily done by precomputing the rewards for each word with each other word in the vocab and 
        ## then look up the reward at training time
        for sp_indices, lt_indices, elem_logits, elem_labels in zip(rhyming_indices, last_token_indices, logits,
                                                                    labels):
            last_token_logits = elem_logits[lt_indices]
            last_token_labels = np.array(elem_labels)[lt_indices]
            first_sentence_indices = sp_indices[:, 0]
            second_sentence_indices = sp_indices[:, 1]
            gold_actions = np.array(last_token_labels)[
                first_sentence_indices]  # get gold for first reference sentence (this because it has seen this token at training time - teacher forcing-)
            second_logits = last_token_logits[second_sentence_indices]  # get the logits for the token that has to rhyme
            predicted_actions, scores = self.__sample(second_logits, num_samples)
            gold_actions_str = [self.tokenizer.decode(x).strip() for x in gold_actions]
            predicted_actions_str = [[self.tokenizer.decode(x).strip() for x in arr] for arr in predicted_actions]
            predictions_info.extend(list(zip(gold_actions_str, predicted_actions_str, scores.tolist())))

        rewards, action_scores = self.__compute_rewards(predictions_info)
        rewards = torch.Tensor(rewards)
        return torch.mean(-torch.Tensor(action_scores) * rewards).to(logits.device), rewards.to(
            logits.device)  # negative because we want to maximise the reward

    def __get_rhyming_indices(self, schemas):
        all_indices = []
        for schema in schemas:
            indices = []
            for i in range(len(schema)):
                for j in range(i + 1, len(schema)):
                    if schema[i] == schema[j]:
                        indices.append((i, j))
            all_indices.append(np.array(indices))
        return all_indices

    def __sample(self, logits, num_samples=1):
        m = Categorical(logits=logits)
        sampled_actions = None
        sampled_scores = None
        for _ in range(num_samples):
            action = m.sample()
            scores = m.log_prob(action)
            if sampled_actions is None:
                sampled_actions = action
                sampled_scores = scores
            else:
                sampled_actions = torch.concat([sampled_actions, action], 0)
                sampled_scores = torch.concat([sampled_scores, scores], 0)
        return sampled_actions.view(logits.shape[0], -1), sampled_scores.view(logits.shape[0], -1)

    def __do_rhyme(self, a, b):
        if a.lower() == b.lower():
            return True
        a_phones = pronouncing.phones_for_word(a)
        b_phones = pronouncing.phones_for_word(b)
        if len(a_phones) < 1:
            return False
        if len(b_phones) < 1:
            return False
        a_phones = a_phones[0]
        b_phones = b_phones[0]
        a_rhyming_phones = pronouncing.rhyming_part(a_phones)
        b_rhyming_phones = pronouncing.rhyming_part(b_phones)
        return a_rhyming_phones == b_rhyming_phones

    def __compute_rewards(self, prediction_info):
        all_rewards = []
        all_scores = []
        for gold, preds, scores in prediction_info:
            local_rewards = []
            local_scores = []
            for pred, score in zip(preds, scores):
                if self.__do_rhyme(gold, pred):
                    local_rewards.append(1.0)
                else:
                    local_rewards.append(-1.0)

                local_scores.append(score)
            avg_reward = sum(local_rewards) / len(local_rewards)
            avg_score = sum(local_scores) / len(local_scores)
            all_rewards.append(avg_reward)
            all_scores.append(avg_score)
        return np.array(all_rewards), np.array(all_scores)
