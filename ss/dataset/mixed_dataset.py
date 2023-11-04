from typing import List
import os
from glob import glob

import torch
import torchaudio
from torch.utils.data import Dataset

from ss.utils.parse_config import ConfigParser
from ss.mixer import MixtureGenerator
from ss.dataset.speaker_collector import LibriSpeechSpeakerFiles

class LibriSpeechMixedDataset(Dataset):
    def __init__(
            self,
            split: str,
            speakers_dataset: str, 
            path_mixtures: str,  
            snr_levels: List[int], 
            update_steps: int,
            trim_db: int,
            vad_db: int,
            audio_len: int,
            n_mixes: int,
            audio_template: str = '*.flac',
            premixed: bool = False,
            config_parser: ConfigParser = None):
        
        if premixed == False:
            speaker_ids = [speaker.name for speaker in os.scandir(speakers_dataset)]
            speakers_files = [LibriSpeechSpeakerFiles(id, speakers_dataset, audio_template=audio_template) for id in speaker_ids]
            self.mix_generator = MixtureGenerator(
                speakers_files=speakers_files,
                save_mixes_to=path_mixtures,
                n_files=n_mixes,
                test= (split != 'train')
            )

            self.mix_generator.generate_mixers(
                snr_levels=snr_levels,
                num_workers=2,
                update_steps=update_steps,
                trim_db=trim_db,
                vad_db=vad_db,
                audio_len=audio_len
            )

        self.reference_files = sorted(glob(os.path.join(path_mixtures, '*-ref.wav')))
        self.mixes_files = sorted(glob(os.path.join(path_mixtures, '*-mixed.wav')))
        self.target_files = sorted(glob(os.path.join(path_mixtures, '*-target.wav')))

        self.config_parser = config_parser

        assert len(self.reference_files) == len(self.mixes_files) and len(self.mixes_files) == len(self.target_files)
    
    def __len__(self):
        return len(self.reference_files)
    
    def _extract_ids(self, mix_path):
        # return order: target_id, noise_id
        filename = os.path.split(mix_path)[1]
        target_id = filename.split('_')[0]
        noise_id = filename.split('_')[1]
        return target_id, noise_id
    
    def _load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        if self.config_parser is not None:
            target_sr = self.config_parser["preprocessing"]["sr"]
            if sr != target_sr:
                audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor
    
    def __getitem__(self, item):
        # TODO: extract noise and target id
        target_id, noise_id = self._extract_ids(self.mixes_files[item])
        ref_audio = self._load_audio(self.reference_files[item])
        mix_audio = self._load_audio(self.mixes_files[item])
        target_audio = self._load_audio(self.target_files[item])
        return {
            'reference': ref_audio,
            'mix': mix_audio,
            'target': target_audio,
            'target_id': target_id,
            'noise_id': noise_id
        }