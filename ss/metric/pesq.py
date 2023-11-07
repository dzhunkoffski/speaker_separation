from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from ss.base.base_metric import BaseMetric

class PESQ(BaseMetric):
    def __init__(self, sampling_frequency: int, mode: str, epoch_freq: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_freq = epoch_freq
        self.last_value = 0
        self.pesq = PerceptualEvaluationSpeechQuality(
            fs=sampling_frequency, mode=mode
        )

    def __call__(self, target, s1, epoch: int, **kwargs):
        if epoch % self.epoch_freq == 0:
            self.last_value = self.pesq(s1, target)
        return self.last_value