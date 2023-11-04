from typing import Any
import torch
from typing import Dict
from torch import nn
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
    
class SpexLoss(nn.Module):
    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.short_sisdr = ScaleInvariantSignalDistortionRatio()
        self.middle_sisdr = ScaleInvariantSignalDistortionRatio()
        self.long_sisdr = ScaleInvariantSignalDistortionRatio()
        # TODO: CrossEntropy

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor, target: torch.Tensor, **batch):
        a = (1 - self.alpha - self.beta) * self.short_sisdr(s1, target)
        b = self.alpha * self.middle_sisdr(s2, target)
        c = self.beta * self.long_sisdr(s3, target)
        return -(a + b + c)