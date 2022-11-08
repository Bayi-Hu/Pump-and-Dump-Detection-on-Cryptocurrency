
# We are organizing this repo to update the code and dataset. (2022/11/07)

# Sequence-based Target Coin Prediction for Cryptocurrency Pump-and-Dump (SIGMOD 2023)

This is the repo including dataset and source code used in the paper ["Sequence-based Target Coin Prediction for Cryptocurrency Pump-and-Dump"](https://arxiv.org/pdf/2204.12929.pdf).

<!-- <div align=center><img width="360" height="250" src="https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/blob/master/materials/showcase.png"/></div> -->


### Data Science Pipeline

The workflow mainly consists of two parts: data collection and target coin prediction. 

<div align=center><img width="680" height="270" src="https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/blob/master/materials/workflow.png"/></div>

Data collection corresponds to the "0_TelegramData" fold, and target coin prediction corresponds to the "1_Statistics" and "TargetCoinPrediction" folds.

### Dataset

The [P&D dataset](https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/blob/master/0_TelegramData/Labeled/pump_attack_new.txt) includes 1,335 samples and 709 P&Ds that we observed on Telegram from January 1, 2019 to January 15, 2022. We will continuously update this dataset.


<!-- ### Model

<div align=center><img width="400" height="300" src="https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/blob/master/materials/SNN.png"/></div> -->


## Geting Start

### Requirements

* Python >= 3.6.1
* NumPy >= 1.12.1
* TensorFlow >= 1.4.0

### Download the Dataset

Due to the 100MB limitation on a single file, we upload our contructed features on the Google Drive.

* Step 1: Download the amazon product dataset of electronics category, which has 498,196 products and 7,824,482 records, and extract it to `raw_data/` folder.
```sh
mkdir raw_data/;
cd utils;
bash 0_download_raw.sh;
```
* Step 2: Convert raw data to pandas dataframe, and remap categorical id.
```sh
python 1_convert_pd.py;
python 2_remap_id.py
```

### Preprocess


### Training and Evaluation



## Citation

If you find this repo useful, please consider to cite our paper:

```
@article{hu2022sequence,
  title={Sequence-Based Target Coin Prediction for Cryptocurrency Pump-and-Dump},
  author={Hu, Sihao and Zhang, Zhen and Lu, Shengliang and He, Bingsheng and Li, Zhao},
  journal={arXiv preprint arXiv:2204.12929},
  year={2022}
}

```





