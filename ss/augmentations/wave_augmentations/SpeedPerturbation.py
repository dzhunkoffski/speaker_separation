from torch import Tensor
import torchaudio
import random

from ss.augmentations.base import AugmentationBase

class SpeedPerturbation(AugmentationBase):
    def __init__(self, p: float, *args, **kwargs):
        self._aug = torchaudio.transforms.SpeedPerturbation(*args, **kwargs)
        self.p = p
    
    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = data.unsqueeze(1)
            x, _ = self._aug(x)
            return x.squeeze(1)
        return data