import torch
from torch import nn
from ss.base.base_metric import BaseMetric

class CrossEntropy(BaseMetric):
    def __init__(self, epoch_freq: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.epoch_freq = epoch_freq
        self.last_value = 100

    def __call__(self, target_id: torch.tensor, epoch: int, sp_logits = None, **batch):
        if sp_logits is not None and epoch % self.epoch_freq == 0:
            self.last_value = self.loss(sp_logits, target_id)
        return self.last_value