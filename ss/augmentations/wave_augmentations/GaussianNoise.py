import torch
from torch import Tensor
from ss.augmentations.base import AugmentationBase
import random

class GaussianNoise(AugmentationBase):
    def __init__(self, p: float, alpha: float, mean, std, **kwargs):
        self.noiser = torch.distributions.normal.Normal(mean, std)
        self.p = p
        self.alpha = alpha
    
    def __call__(self, data: Tensor) -> Tensor:
        if random.random() < self.p:
            x = data.unsqueeze(1)
            x = x + self.alpha * self.noiser.sample(x.shape)
            x = x.squeeze(1)
            return x
        return data