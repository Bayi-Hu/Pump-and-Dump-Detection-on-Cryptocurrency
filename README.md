
# I am organizing this repo to update the code and dataset. (2022/11/07)

----------------------
# Target Coin Prediction for Cryptocurrency Pump-and-Dump


This is the dataset and source code used in the paper "Target Coin Prediction for Cryptocurrency Pump-and-Dump".

<!-- <div align=center><img width="360" height="250" src="https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/blob/master/materials/showcase.png"/></div> -->


### Workflow

The workflow mainly consists of two parts: data collection and target coin prediction. 

<div align=center><img width="680" height="270" src="https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/blob/master/materials/workflow.png"/></div>

Data collection corresponds to the "0_TelegramData" fold, and target coin prediction corresponds to the "1_Statistics" and "TargetCoinPrediction" folds.

### Dataset

The [P&D dataset](https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/blob/master/0_TelegramData/Labeled/pump_attack_new.txt) includes 1,335 samples and 709 P&Ds that we observed on Telegram from January 1, 2019 to January 15, 2022. We will continuously update this dataset.

### Results

**Pump Message Detection**
| Model | AUC | Precision | Recall| F1|
| ------  | ------ |----| ---|---|
|LR  |0.988| 0.892| 0.913|0.902|
|RF |0.994| 0.901| 0.939|0.920|


**Target Coin Prediction**

| Metrics | Random | LR | DNN| SNNv|SNN|
| ------  | ------ |----| ---|---|---|
|AUC  |0.500| 0.888| 0.920|0.927|0.935|
|HR@1 |0.003| 0.145| 0.225|0.233|0.237|
|HR@3 |0.009| 0.255| 0.278|0.339|0.392|
|HR@5 |0.016| 0.291| 0.383|0.432|0.495|
|HR@10|0.031| 0.357| 0.498|0.558|0.596|
|HR@20|0.062| 0.581| 0.626|0.652|0.695|
|HR@30|0.093| 0.635| 0.727|0.740|0.749|


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

This implementation not only contains the SNN method, but also provides other competitors' methods, including DNN and SNNv. The training procedures of all method is as follows:


```
python experiments.py --model=simple-cnn \
    --dataset=cifar10 \
    --alg=fedprox \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=10 \
    --n_parties=10 \
    --mu=0.01 \
    --rho=0.9 \
    --comm_round=50 \
    --partition=noniid-labeldir \
    --beta=0.5\
    --device='cuda:0'\
    --datadir='./data/' \
    --logdir='./logs/' \
    --noise=0 \
    --sample=1 \
    --init_seed=0
```

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `simple-cnn`, `vgg`, `resnet`, `mlp`. Default = `mlp`. |
| `max_length` | The maximum ength of the sequence, ranging from 1 to 50, default=20. |
| `lr` | Learning rate for the local models, default = `0.01`. |
| `batch-size` | Batch size, default = `64`. |
| `epochs` | Number of local training epochs, default = `5`. |
| `comm_round`    | Number of communication rounds to use, default = `50`. |
| `partition`    | The partition way. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns), `real`, `iid-diff-quantity`. Default = `homo` |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`. |
| `device` | Specify the device to run the program, default = `cuda:0`. |
| `datadir` | The path of the dataset, default = `./data/`. |
| `logdir` | The path to store the logs, default = `./logs/`. |
| `noise` | Maximum variance of Gaussian noise we add to local party, default = `0`. |
| `sample` | Ratio of parties that participate in each communication round, default = `1`. |
| `init_seed` | The initial seed, default = `0`. |


