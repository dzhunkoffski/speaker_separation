import argparse
import glob
import random
import os

from ss.mixer.mixture_generator import create_mix
from tqdm.auto import tqdm

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Wham Mixer')
    args.add_argument(
        "-s",
        "--speakers",
        default=None,
        type=str,
        help="path to folder with speakers"
    )
    args.add_argument(
        '-n',
        '--noise',
        default=None,
        type=str,
        help="path to WHAM noise",
    )
    args.add_argument(
        '-o',
        '--out',
        default=None,
        type=str,
        help="destination to save mixes"
    )
    args = args.parse_args()

    speaker_ids = ['1995', '61']
    speaker1_audios = glob.glob(f'{args.speakers}/{speaker_ids[0]}/**/*.flac', recursive=True)
    speaker2_audios = glob.glob(f'{args.speakers}/{speaker_ids[1]}/**/*.flac', recursive=True)
    speakers = [speaker1_audios, speaker2_audios]
    noise_audios = glob.glob(f'{args.noise}/**/*.wav', recursive=True)
    triplets = []
    for i in range(1000):
        speaker_num = random.choice([0, 1])
        target, ref = random.sample(speakers[speaker_num], 2)
        noise = random.choice(noise_audios)

        triplet = {
            'reference': ref,
            'target': target,
            'noise': noise,
            'target_id': speaker_num,
            'noise_id': i
        }
        triplets.append(triplet)

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    for i in tqdm(range(1000)):
        # сделал snr чуть ниже 0, так как очень тихие спикеры
        create_mix(idx=i, triplet=triplets[i], snr_levels=[-0.13], out_dir=args.out, test=True, sr=16000, trim_db=20, vad_db=20, audio_len=4)