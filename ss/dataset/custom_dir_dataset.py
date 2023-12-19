from typing import List
import os
from glob import glob

import torch
import torchaudio
from torch.utils.data import Dataset

from ss.utils.parse_config import ConfigParser
from ss.mixer import MixtureGenerator
from ss.dataset.speaker_collector import LibriSpeechSpeakerFiles

class CustomDirDataset(Dataset):
    def __init__(self, mixes_dir: str, refs_dir: str, targets_dir: str, **args):
        self.mixes = sorted(glob(f'{mixes_dir}/**-mixed.wav'))
        self.refs = sorted(glob(f'{refs_dir}/**-ref.wav'))
        self.targets = sorted(glob(f'{targets_dir}/**-target.wav'))

        assert len(self.mixes) == len(self.refs) and len(self.refs) == len(self.targets)

    def __len__(self):
        return len(self.mixes)
    
    def _extract_id(self, path):
        filename = os.path.split(path)[1]
        id = filename.split('-')[0]
        return id
    
    def _load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        target_sr = 16000
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor
    
    def __getitem__(self, item):
        id = self._extract_id(self.refs[item])
        ref_audio = self._load_audio(self.refs[item])
        mix_audio = self._load_audio(self.mixes[item])
        target_audio = self._load_audio(self.targets[item])

        return {
            'mix_path': self.mixes[item],
            'reference': ref_audio,
            'mix': mix_audio,
            'target': target_audio,
            'target_id': id,
            'noise_id': id,
        }