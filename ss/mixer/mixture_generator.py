import random
import os
from concurrent.futures import ProcessPoolExecutor
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
import librosa

def snr_mixer(clean, noise, snr):
    amp_noise = np.linalg.norm(clean) / 10**(snr / 20)
    noise_norm = (noise / np.linalg.norm(noise)) * amp_noise
    mix = clean + noise_norm
    return mix

def vad_merge(w, top_db):
    intervals = librosa.effects.split(w, top_db=top_db)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)

def cut_audios(s1, s2, sec, sr):
    cut_len = sr * sec
    len1 = len(s1)
    len2 = len(s2)

    s1_cut = []
    s2_cut = []

    segment = 0
    while (segment + 1) * cut_len < len1 and (segment + 1) * cut_len < len2:
        s1_cut.append(s1[segment * cut_len:(segment + 1) * cut_len])
        s2_cut.append(s2[segment * cut_len:(segment + 1) * cut_len])
        segment += 1
    return s1_cut, s2_cut

def fix_length(s1, s2, min_or_max='max'):
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:
        utt_len = np.maximum(len(s1), len(s2))
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
    return s1, s2

def create_mix(idx, triplet, snr_levels, out_dir, test=False, sr=16000, **kwargs):
    trim_db, vad_db = kwargs['trim_db'], kwargs['vad_db']
    audio_len = kwargs['audio_len']

    s1_path = triplet['target']
    s2_path = triplet['noise']
    ref_path = triplet['reference']
    target_id = triplet['target_id']
    noise_id = triplet['noise_id']

    s1, _ = sf.read(os.path.join('', s1_path))
    s2, _ = sf.read(os.path.join('', s2_path))
    if len(s2.shape) > 1:
        s2 = s2[:, 0]
    ref, _ = sf.read(os.path.join('', ref_path))

    meter = pyln.Meter(sr)
    louds1 = meter.integrated_loudness(s1)
    louds2 = meter.integrated_loudness(s2)
    louds_ref = meter.integrated_loudness(ref)

    s1_norm = pyln.normalize.loudness(s1, louds1, -29)
    s2_norm = pyln.normalize.loudness(s2, louds2, -29)
    ref_norm = pyln.normalize.loudness(ref, louds_ref, -23.0)

    s1_amp = np.max(np.abs(s1_norm))
    s2_amp = np.max(np.abs(s2_norm))
    ref_amp = np.max(np.abs(ref_norm))

    if s1_amp == 0 or s2_amp == 0 or  ref_amp == 0:
        return
    
    if trim_db:
        ref, _ = librosa.effects.trim(ref_norm, top_db=trim_db)
        s1, _ = librosa.effects.trim(s1_norm, top_db=trim_db)
        s2, _ = librosa.effects.trim(s2_norm, top_db=trim_db)

    if len(ref) < sr:
        return
    
    path_mix = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-mixed.wav")
    path_target = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-target.wav")
    path_ref = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-ref.wav")

    snr = np.random.choice(snr_levels, 1).item()

    if not test:
        s1, s2 = vad_merge(s1, vad_db), vad_merge(s2, vad_db)
        s1_cut, s2_cut = cut_audios(s1, s2, audio_len, sr)
        for i in range(len(s1_cut)):
            mix = snr_mixer(s1_cut[i], s2_cut[i], snr)
            louds1 = meter.integrated_loudness(s1_cut[i])
            s1_cut[i] = pyln.normalize.loudness(s1_cut[i], louds1, -23.0)
            loud_mix = meter.integrated_loudness(mix)
            mix = pyln.normalize.loudness(mix, loud_mix, -23.0)

            path_mix_i = path_mix.replace("-mixed.wav", f"_{i}-mixed.wav")
            path_target_i = path_target.replace("-target.wav", f"_{i}-target.wav")
            path_ref_i = path_ref.replace("-ref.wav", f"_{i}-ref.wav")

            sf.write(path_mix_i, mix, sr)
            sf.write(path_target_i, s1_cut[i], sr)
            sf.write(path_ref_i, ref, sr)
    else:
        s1, s2 = fix_length(s1, s2, 'max')
        mix = snr_mixer(s1, s2, snr)
        louds1 = meter.integrated_loudness(s1)
        s1 = pyln.normalize.loudness(s1, louds1, -23.0)
        loud_mix = meter.integrated_loudness(mix)
        mix = pyln.normalize.loudness(mix, loud_mix, -23.0)
        sf.write(path_mix, mix, sr)
        sf.write(path_target, s1, sr)
        sf.write(path_ref, ref, sr)

class MixtureGenerator:
    def __init__(self, speakers_files, save_mixes_to: str, n_files=5000, test=False, random_state=42):
        self.speakers_files = speakers_files
        self.n_files = n_files
        self.random_state = random_state
        self.save_mixes_to = save_mixes_to
        self.test = test
        random.seed(self.random_state)
        if not os.path.exists(self.save_mixes_to):
            os.makedirs(self.save_mixes_to)
        
    def generate_triplets(self):
        i = 0
        all_triplets = {
            'reference': [],
            'target': [],
            'noise': [],
            'target_id': [],
            'noise_id': []
        }
        while i < self.n_files:
            spk1, spk2 = random.sample(self.speakers_files, 2)
            if len(spk1.files) < 2 or len(spk2.files) < 2:
                continue
            target, reference = random.sample(spk1.files, 2)
            noise = random.choice(spk2.files)

            all_triplets['reference'].append(reference)
            all_triplets['target'].append(target)
            all_triplets['noise'].append(noise)
            all_triplets['target_id'].append(spk1.id)
            all_triplets['noise_id'].append(spk2.id)
            i += 1

        return all_triplets
    
    def generate_mixers(self, snr_levels=[0], num_workers=10, update_steps=10, **kwargs):
        triplets = self.generate_triplets()
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = []
            for i in range(self.n_files):
                triplet = {
                    'reference': triplets['reference'][i],
                    'target': triplets['target'][i],
                    'noise': triplets['noise'][i],
                    'target_id': triplets['target_id'][i],
                    'noise_id': triplets['noise_id'][i]
                }
                futures.append(pool.submit(create_mix, i, triplet, snr_levels, self.save_mixes_to, test=self.test, **kwargs))
            for i, future in enumerate(futures):
                future.result()