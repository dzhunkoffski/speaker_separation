import logging
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    mix_audio_batch = []
    ref_audio_batch = []
    target_audio_batch = []
    mix_audio_path_batch = []
    target_id_batch = []
    noise_id_batch = []
    max_audio_len = 0
    for item in dataset_items:
        mix_audio_batch.append(item['mix'].t())
        ref_audio_batch.append(item['reference'].t())
        target_audio_batch.append(item['target'].t())
        mix_audio_path_batch.append(item['mix_path'])
        target_id_batch.append(item['target_id'])
        noise_id_batch.append(item['noise_id'])

    mix_audio_batch = torch.permute(pad_sequence(mix_audio_batch, batch_first=True, padding_value=0), (0, 2, 1))
    ref_audio_batch = torch.permute(pad_sequence(ref_audio_batch, batch_first=True, padding_value=0), (0, 2, 1))
    target_audio_batch = torch.permute(pad_sequence(target_audio_batch, batch_first=True, padding_value=0), (0, 2, 1))
    target_id_batch = torch.tensor(target_id_batch)

    return {
        'mix_path': mix_audio_path_batch,
        'reference': ref_audio_batch,
        'mix': mix_audio_batch,
        'target': target_audio_batch,
        'target_id': target_id_batch,
        'noise_id': noise_id_batch
    }