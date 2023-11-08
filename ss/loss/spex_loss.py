from typing import Any
import torch
from typing import Dict
from torch import nn
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
    
class SpexLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.short_sisdr = ScaleInvariantSignalDistortionRatio()
        self.middle_sisdr = ScaleInvariantSignalDistortionRatio()
        self.long_sisdr = ScaleInvariantSignalDistortionRatio()

        self.clf_loss = nn.CrossEntropyLoss()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor, target: torch.Tensor, target_id: torch.Tensor, sp_logits = None, **batch):
        a = (1 - self.alpha - self.beta) * self.short_sisdr(s1, target)
        b = self.alpha * self.middle_sisdr(s2, target)
        c = self.beta * self.long_sisdr(s3, target)
        loss = -(a + b + c)
        if sp_logits is not None:
            loss = loss + self.gamma * self.clf_loss(sp_logits, target_id) 
        return loss
        

        
