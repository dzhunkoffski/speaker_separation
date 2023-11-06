import torch
import torchaudio
from torch import Tensor
from ss.augmentations.base import AugmentationBase
import random

class PitchShift(AugmentationBase):
    def __init__(self, p: float, sample_rate, n_steps, **kwargs):
        self._aug = torchaudio.transforms.PitchShift(sample_rate, n_steps, **kwargs)
        self.p = p
    
    def __call__(self, data: Tensor) -> Tensor:
        if random.random() < self.p:
            x = data.unsqueeze(1)
            x = self._aug(x)
            return x.squeeze(1)
        return data