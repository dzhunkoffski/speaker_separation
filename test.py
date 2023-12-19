import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import ss.model as module_model
from ss.trainer import Trainer
from ss.utils import ROOT_PATH
from ss.utils.object_loading import get_dataloaders
from ss.utils.parse_config import ConfigParser

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
import torchaudio

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    config['arch']['args']['n_speakers'] = 143
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    sisdr = ScaleInvariantSignalDistortionRatio().to(device)
    pesq = PerceptualEvaluationSpeechQuality(
            fs=16000, mode='wb'
    )

    sisdr_metric = 0
    pesq_metric = 0

    with torch.no_grad():
        for item in tqdm(dataloaders['test'].dataset):
            mix = item['mix'].to(device).unsqueeze(0)
            reference = item['reference'].to(device).unsqueeze(0)
            target = item['target'].to(device).unsqueeze(0)
            pred = model(mix=mix, reference=reference, is_train=False)

            sisdr_metric += sisdr(pred['s1'], target).item()
            pesq_metric += pesq(pred['s1'], target).item()
            if out_file is not None:
                if not os.path.exists(out_file):
                    os.makedirs(out_file)
                
                wav_name = f'{os.path.split(item["mix_path"])[-1].split("-")[0]}-predicted.flac'
                torchaudio.save(
                    uri=f"{out_file}/{wav_name}",
                    src=pred['s1'].squeeze(0).cpu(),
                    sample_rate=config['preprocessing']['sr']
                )
    
    sisdr_metric /= len(dataloaders['test'].dataset)
    pesq_metric /= len(dataloaders['test'].dataset)

    print('SISDR:', sisdr_metric)
    print('PESQ:', pesq_metric)




if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="directory to write predictions",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirDataset",
                        "args": {
                            "mixes_dir": str(test_data_folder),
                            "refs_dir": str(
                                test_data_folder
                            ),
                            "targets_dir": str(
                                test_data_folder
                            )
                        },
                    }
                ],
            }
        }

    print('Instantiated customdirdataset')

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
