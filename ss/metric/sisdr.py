from ss.base.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

class EvalSISDR(BaseMetric):
    def __init__(self, epoch_freq: int,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = ScaleInvariantSignalDistortionRatio()
        self.epoch_freq = epoch_freq
        self.last_value = -100
    
    def __call__(self, s1, target, epoch, **batch):
        if epoch % self.epoch_freq == 0:
            self.loss = self.loss.to(s1.get_device())
            self.last_value = self.loss(s1, target)
        return self.last_value