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


### P&D Data Collection

The pipleline of P&D data collection from Telegram:

```
cd 0_TelegramData
python 0_get_channel_post.py          # You need to fill in your own Telegram api_id and api_hash. You can apply in https://core.telegram.org
python 1_keyword_filtering.py
python 2_pump_message_labeling.py     # We have alreadly labeled 5000+ samples in ./Labeled/label.txt
python 3_message_fg.py 
python 4_classifier_training.py 
python 5_classifier_prediction.py
python 6_session_aggregation.py
python 7_P&D_labeling.py              
python 8_data_cleaning.py             # Generate the final P&D samples in ./Labeled/pump_attack_new.txt
```


### Feature Generation

There are two ways for feature generation:

#### 1. Generate feature from the raw P&D dataset

* Step 1: Download the historical statistics.

```
cd 1_Statistics
python 0_kline_collect_Binance.py
python 1_data_process.py
```

* Step 2: Generate features for positive and negative samples respectively.

```
cd TargetCoinPrediction/FeatureGeneration
python feature_generation.py
python pos_sample_process.py
python neg_sample_process.py
```

#### 2. Download contructed features from our Google Drive



### Target Coin Prediction

#### Training

This implementation not only contains the SNN method, but also provides other competitors' methods, including DNN and SNNv. The training procedures of all method is as follows:

```
cd TargetCoinPrediction/Train
```

```
python train_model_seq_pos_atten.py
    --lr=0.01 \
    --batch-size=256 \
    --epochs=20 \
    --n_parties=10 \
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


#### Evaluation

```
python eval_model_seq_pos_atten.py
    --lr=0.01 \
    --batch-size=256 \
    --epochs=20 \
    --n_parties=10 \
    --comm_round=50 \
    --device='cuda:0'\
    --datadir='./data/' \
    --logdir='./logs/' \
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
| `device` | Specify the device to run the program, default = `cuda:0`. |
| `datadir` | The path of the dataset, default = `./data/`. |
| `logdir` | The path to store the logs, default = `./logs/`. |
| `noise` | Maximum variance of Gaussian noise we add to local party, default = `0`. |
| `sample` | Ratio of parties that participate in each communication round, default = `1`. |
| `init_seed` | The initial seed, default = `0`. |





