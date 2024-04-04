from typing import Any
import pytorch_lightning as pl
from torchmetrics import Accuracy

from src.robing.rhyming_relation_probe import RhymingProbe
import torch

class RhymingProbeModule(pl.LightningModule):
   def __init__(self, conf, *args, **kwargs) -> None:
      super().__init__()
      self.save_hyperparameters(conf)
      self.model = RhymingProbe(conf.model.hidden_size, conf.model.activation_fun if 'activation_fun' in conf.model else None)
      self.val_macro_accuracy_metric = Accuracy(threshold=0.5, num_classes=2, average='macro')
      self.val_micro_accuracy_metric = Accuracy(threshold=0.5, num_classes=2, average='micro')
      self.test_macro_accuracy_metric = Accuracy(threshold=0.5, num_classes=2, average='macro')
      self.test_micro_accuracy_metric = Accuracy(threshold=0.5, num_classes=2, average='micro')


   def on_validation_epoch_start(self) -> None:
      self.val_macro_accuracy_metric = Accuracy(threshold=0.5, num_classes=2, average='macro')
      self.val_micro_accuracy_metric = Accuracy(threshold=0.5, num_classes=2, average='micro')

   def on_test_epoch_start(self) -> None:
      self.test_macro_accuracy_metric = Accuracy(threshold=0.5, num_classes=2, average='macro')
      self.test_micro_accuracy_metric = Accuracy(threshold=0.5, num_classes=2, average='micro')

   def forward(self, batch) -> dict:
      return self.model(batch['input_embeddings'], batch.get('labels'))
   
   def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
      forward_output = self.forward(batch)
      self.log("loss", forward_output["loss"], sync_dist=True)
      return forward_output["loss"]
   
   def validation_step(self, batch: dict, batch_idx: int):
      forward_output = self.forward(batch)
      self.log("val_loss", forward_output["loss"], sync_dist=True)
      self.val_micro_accuracy_metric.update(forward_output['logits'], batch['labels'])
      self.val_macro_accuracy_metric.update(forward_output['logits'], batch['labels'])
      return forward_output['loss']
   
   def test_step(self, batch: dict, batch_idx: int):
      forward_output = self.forward(batch)
      self.test_micro_accuracy_metric.update(forward_output['logits'], batch['labels'])
      self.test_macro_accuracy_metric.update(forward_output['logits'], batch['labels'])
      self.log("test_loss", forward_output["loss"], sync_dist=True)
      
      return forward_output['loss']

   def __get_predictions(self, logits):
      return (torch.sigmoid(logits) >= .5).type(torch.int)

   def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
      forward_output = self.forward(batch)
      forward_output['predictions'] = self.__get_predictions(forward_output['logits'])
      return forward_output

   def validation_epoch_end(self, outputs) -> None:
      micro_accuracy = self.val_micro_accuracy_metric.compute()
      macro_accuracy = self.val_micro_accuracy_metric.compute()
      self.log('val_micro_acc', micro_accuracy)
      self.log('val_macro_acc', macro_accuracy)
      

   def test_epoch_end(self, outputs) -> None:
      micro_accuracy = self.test_micro_accuracy_metric.compute()
      macro_accuracy = self.test_micro_accuracy_metric.compute()
      self.log('test_micro_acc', micro_accuracy)
      self.log('test_macro_acc', macro_accuracy)
      
   
   

