

# Sequence-based Target Coin Prediction for Cryptocurrency Pump-and-Dump (SIGMOD 2023)

This is the repo including dataset and code used in the paper ["Sequence-based Target Coin Prediction for Cryptocurrency Pump-and-Dump"](https://arxiv.org/pdf/2204.12929.pdf).

<!-- <div align=center><img width="360" height="250" src="https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/blob/master/materials/showcase.png"/></div> -->


### Data Science Pipeline

The workflow mainly consists of two parts: data collection and target coin prediction. 

<div align=center><img width="680" height="270" src="https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/blob/master/materials/workflow.png"/></div>


### Pump-and-dump Activity Logs (Jan. 1, 2019 to Jan. 15, 2021) 

The [P&D logs](https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/blob/master/Data/Telegram/Labeled/PD_log_cleaned.txt) includes 1,335 samples and 709 P&Ds that we observed on Telegram. 
We will periodically update this dataset.

<!-- ### SeqModel

<div align=center><img width="400" height="300" src="https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/blob/master/materials/SNN.png"/></div> -->

# Getting Start

Since we have already collected P\&D log dataset and will periodically update it, you can skip the data collection part : )

##  0. Data Collection

First, we get seed channels verified by PumpOlymp and explore the pump channels.

- /Data/Telegram/Pump_channel1
- /Data/Telegram/Pump_channel2

#### Step1: get historical messages according to channel
``` 
python get_channel_post.py
```
#### Step2: Select pump message with keyword filtering
``` 
python keyword_filte.py
```
#### Step3: Manually label the filtered messages
``` 
python pump_message_label.py
# we have already labeled 5000+ samples. 
```

#### Step4: Generate features for messages.
``` 
python message_fg.py
```

#### Step5: Train a detection classifier.
``` 
python train_classifier.py
```

#### Step6: Use the classifier for detection 
``` 
python predict_classifier.py
```

#### Step7: Aggregate the session based on timestamp 
``` 
python sess_aggregate.py
```

#### Step8: Label the predicted pump session and generate final P&D log
``` 
python PD_label.py
```

#### Step9: Clean the log dataset
``` 
python PDlog_clean.py
```


## 1. Target Coin Prediction

### 1.1 Feature Generation:

Two methods to generate features for Target Coin Prediction.

#### Method1: Generate features from P&D logs: 

```
HOLD (still organizing this part of code because it involves feature collection from multiple sources,)
``` 

#### Mehod2: Download generated dataset from Good Drive

* [train_sample](https://drive.google.com/file/d/1u2Ichky12k-ZTHDhqgFLM5WzlH26JnKa/view?usp=sharing)

* [test_sample](https://drive.google.com/file/d/1slLs-OqMqzLHrmvzbf8xlyP2zzDpIk1R/view?usp=sharing)

``` 
cd TargetCoinPrediction
tar -xzvf train.tar.gz
tar -xzvf test.tar.gz
``` 

### 1.2 Model Training

#### Step1: Train SNN model
``` 
cd TargetCoinPrediction/SeqModel
python run_train.py  --model=snn \
                     --max_seq_length=8 \
                     --epoch=5 \
                     --batch_size=256 \
                     --learning_rate=1e-4 \
                     --dropout_rate=0.2
                     --do_train=True
                     --do_eval=False \
                     --checkpointDir=xxx \
                     --init_seed=1234 
```


| Parameter        | Description                                                            |
|------------------|------------------------------------------------------------------------|
| `model`          | Model used for target coin prediction, options (`snn`, `snnta`, `dnn`) |
| `max_seq_length` | The maximum length of P&D sequence, options `1~ 50`, default=`8`.      |
| `epochs`         | Number of training epochs, default = `30`.                             |
| `batch_size`     | Batch size, default = `256`.                                           |
| `learning_rate`  | Learning rate for the optimizer (Adam), default = `5e-4`.              |
| `dropout_rate`   | Dropout Ratio for training, default = `0.2`.                           |
| `do_train`       | Whether to do training or testing, default = `True`.                   |
| `do_eval`        | Whether to do training or testing, default = `False`.                  |
| `checkpointDir`  | Specify the directory to save the checkpoints.                         |
| `init_seed`      | The initial seed, default = `1234`.                                    |



#### Step2: Evaluate SNN model
```
python run_eval.py   --model=snn \
                     --max_seq_length=8 \
                     --epoch=1 \
                     --batch_size=256 \
                     --do_train=False
                     --do_eval=True \
                     --checkpointDir=xxx \
                     --init_seed=1234 
```

### Results on current dataset

We periodically update this table according to updated dataset.

| Metric  | DNN   | SNN   | SNN_s |
|---------|-------|-------|-------|
| HR1     | 0.203 | 0.253 | 0.313 |
| HR3     | 0.295 | 0.361 | 0.445 |
| HR5     | 0.431 | 0.471 | 0.498 |
| HR10    | 0.463 | 0.599 | 0.599 |
| HR20    | 0.630 | 0.709 | 0.714 |
| HR30    | 0.797 | 0.824 | 0.846 |

In this dataset we generate only 9 statistical features and coin id for pumped coin in sequence. 
The performance of SNN can be further improved by using more features for sequence.


-----
## Citation

If you find this repo useful, please cite our paper:

```
@article{hu2022sequence,
  title={Sequence-Based Target Coin Prediction for Cryptocurrency Pump-and-Dump},
  author={Hu, Sihao and Zhang, Zhen and Lu, Shengliang and He, Bingsheng and Li, Zhao},
  journal={arXiv preprint arXiv:2204.12929},
  year={2022}
}

```





