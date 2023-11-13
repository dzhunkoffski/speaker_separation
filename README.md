# speaker_separation
DL-AUDIO homework

## Запуск итогового решения:
сначала загрузим датасеты:
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
теперь запускаем обучение:
```bash

```

## Obtain WHAM mixes
```bash
cd data
wget https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
unzip wham_noise.zip
```
У вас должна получиться папка `wham_noise` по пути `data/wham_noise`. Теперь вернитесь в корневую директорию проекта и запустите скрипт `make_wham_dataset.py`:
```bash
python make_wham_dataset.py --speaker data/librispeech/test-clean --noise data/wham_noise/tr --out data/mixes/wham
```
Далее запустите конфиг, который я специально заранее подготовил, для проверки предсказания на wham-like датасете:
```bash
python test.py --config test_configs/wham.json --resume checkpoints/exp3/checkpoint-epoch80.pth
```
В результатах должно получиться что-то вроде:
```
SISDR: 3.987
PESQ: 1.243
```

## ASR бонус (пример локально)
запустить конфиг, чтобы сделать датасет с сохраненным текстом и заготовить ASR датасет.
```bash
python make_asr_dataset.py --input /home/dzhunk/University/dl-audio/asr/data/datasets/librispeech/test-clean_index.json --out /home/dzhunk/University/dl-audio/speaker_separation/data/mixes/asr --preds /home/dzhunk/University/dl-audio/speaker_separation/predictions
```
построим предсказания для этого датасета
```bash
python test.py --config test_configs/asr.json --output /home/dzhunk/University/dl-audio/speaker_separation/predictions --resume checkpoints/exp3/checkpoint-epoch95.pth
```
(здесь важно чтобы пути были глобальные везде)

теперь открываем `asr` проект и из него запускаем следующий конфиг:
```bash
python test.py --config hw_asr/configs/source_separ_test.json --resume checkpoints/checkpoint-epoch40.pth
```
чтобы получить скор
```bash
python evaluate.py --predictions output.json
```
будет что то вроде такого
```
Argmax WER: 1.805463819495858
Argmax CER: 1.310468686383481
BeamSearch WER: 1.8004802733812928
BeamSearch CER: 1.315871575701202
LM BeamSearch WER: 1.5338908061909988
LM BeamSearch CER: 1.1804251157970114
```