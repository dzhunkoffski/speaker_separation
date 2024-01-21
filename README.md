# speaker_separation
DL-AUDIO homework. More details on model, train configuration and results examples located in wandb [report](https://wandb.ai/dzhunkoffski/speaker_separation/reports/Report--Vmlldzo2NTgzODYw)

## Project structure
```bash
.
├── checkpoints     <-- Move here pretrained model checkpoint and corresponding `config.json`.
├── data            <-- Directory with train, test and validation datasets
├── notebooks       <-- Example notebooks
│    └── speaker-separation.ipynb     <-- Example how to use this repository in kaggle.
└── ss              <-- Module with model, trainer, dataloader and etc.
```

## How to run this repository:
First download the datasets:
```bash
cd data
wget https://us.openslr.org/resources/12/test-clean.tar.gz
tar -xf test-clean.tar.gz
mkdir librispeech
mv LibriSpeech/test-clean librispeech/
rm -r LibriSpeech
```
```bash
cd data
wget https://us.openslr.org/resources/12/train-clean-100.tar.gz
tar -xf train-clean-100.tar.gz
mkdir librispeech
mv LibriSpeech/train-clean-100 librispeech/
rm -r LibriSpeech
cd ..
```
Now train the model:
```bash
python train.py --config ss/configs/spexp.json
```
Put file `checkpoint-epoch95.pth` from https://drive.google.com/drive/folders/1qcMY2sN1mK1AYI27BXjoy4bOomjPJ14c?usp=sharing into `checkpoints/exp3/checkpoint-epoch95.pth` and run test :
```bash
python test.py --config test_configs/public.json --resume checkpoints/exp3/checkpoint-epoch95.pth
```
