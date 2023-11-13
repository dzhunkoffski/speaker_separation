import argparse
import glob
import random
import os
import json

from ss.mixer.mixture_generator import create_mix
from tqdm.auto import tqdm

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='asr Mixer')
    args.add_argument(
        '-i',
        '--input',
        default=None,
        type=str,
        help="destination of source asr dataset json file"
    )
    args.add_argument(
        '-o',
        '--out',
        default=None,
        type=str,
        help="destination to save mixes"
    )
    args.add_argument(
        '-p',
        '--preds',
        default=None,
        help="destination to save predicitons"
    )
    args = args.parse_args()
    with open(args.input, 'r') as fd:
        items = json.load(fd)

    speaker_ids = []
    for item in items:
        item_id = item['path'].split(os.sep)[-3]
        speaker_ids.append(item_id)
    i = 0
    pred_items = []
    triplets = []
    while i < 500:
        target_item = random.choice(items)
        target_audio_path = target_item['path']
        pred_items.append(target_item)
        target_id = target_item['path'].split(os.sep)[-3]
        ref_audio_path = random.choice(glob.glob(f'{os.sep.join(target_audio_path.split(os.sep)[:-2])}/**/*.flac', recursive=True))
        noise_id = random.choice(speaker_ids)
        noise_audio_path = random.choice(glob.glob(f'{os.sep.join(target_audio_path.split(os.sep)[:-3])}/{noise_id}/**/*.flac', recursive=True))

        i += 1

        triplet = {
            'reference': ref_audio_path,
            'target': target_audio_path,
            'noise': noise_audio_path,
            'target_id': target_id,
            'noise_id': noise_id
        }
        triplets.append(triplet)

    if not os.path.exists(args.out):
        os.makedirs(args.out) 
    for i in tqdm(range(500)):
        pred_items[i]['path'] = f'{args.preds}/{triplets[i]["target_id"]}_{triplets[i]["noise_id"]}_{"%06d" % i}-predicted.flac'
        create_mix(idx=i, triplet=triplets[i], snr_levels=[0], out_dir=args.out, test=True, sr=16000, trim_db=20, vad_db=20, audio_len=4)
    
    if not os.path.exists(args.preds):
        os.makedirs(args.preds)
    with open(f'{args.preds}/custom_noise_libr.json', 'w') as fd:
        json.dump(pred_items, fd)
    
