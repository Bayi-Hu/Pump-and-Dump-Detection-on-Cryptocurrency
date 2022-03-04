# -*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import pickle as pkl
from datetime import *
import os

if __name__ == '__main__':
    # train/validation/test 要根据时间点分，否则会有leakage
    pos_df = pd.read_csv("pump_sample_raw.csv")
    pos_df = pos_df[pos_df["exchange"] == "binance"]
    pos_df = pos_df[pos_df["pair"] == "BTC"]

    neg_df = pd.read_csv("neg_sample_raw.csv", na_values="None")
    neg_df = neg_df[neg_df["pair"] == "BTC"]

    #其余的使用平均值填充
    for column in ['pre_3d_market_cap_usd','pre_3d_market_cap_btc', 'pre_3d_price_usd', 'pre_3d_price_btc','pre_3d_volume_usd', 'pre_3d_volume_btc', 'pre_3d_twitter_index', 'pre_3d_reddit_index', 'pre_3d_alexa_index']:
        mean_val = pos_df[column].mean()
        pos_df[column].fillna(mean_val, inplace=True)

    for column in ['pre_3d_market_cap_usd','pre_3d_market_cap_btc', 'pre_3d_price_usd', 'pre_3d_price_btc','pre_3d_volume_usd', 'pre_3d_volume_btc', 'pre_3d_twitter_index', 'pre_3d_reddit_index', 'pre_3d_alexa_index']:
        mean_val = neg_df[column].mean()
        neg_df[column].fillna(mean_val, inplace=True)

    price_columns = []
    return_columns = []
    volume_columns = []

    for column in pos_df.columns:
        if "price" in column:
            price_columns.append(column)
        if "return" in column:
            return_columns.append(column)
        if "volume" in column:
            volume_columns.append(column)

    other_columns = ['pre_3d_alexa_index', 'pre_3d_market_cap_btc', 'pre_3d_market_cap_usd', 'pre_3d_reddit_index', 'pre_3d_twitter_index']

    for column in price_columns:
        pos_df[column] = pos_df[column] * (10**5)
        try:
            neg_df[column] = neg_df[column] * (10**5)
        except:
            continue

    for column in volume_columns + other_columns:
        pos_df[column] = np.log2(pos_df[column]+0.1)
        neg_df[column] = np.log2(neg_df[column]+0.1)

    pos_df["label"] = 1
    neg_df["label"] = 0

    # sequence 化
    for idx, row in pos_df.iterrows():
        feature = []
        for column in price_columns + return_columns + volume_columns:
            feature.append(row[column])

        feature_str = "".join(map(lambda x: str(x), feature))
        pos_df.loc[idx, "feature"] = feature_str

    # pre_feature_columns = []
    X_pos = pd.merge(left=pos_df, right=pos_df[["channel_id", "coin", "feature", "timestamp_unix"]], how='left', on=["channel_id"], sort=False)
    X_pos = X_pos[X_pos.timestamp_unix_x > X_pos.timestamp_unix_y]
    X_pos = X_pos.rename(columns={"timestamp_unix_x": "timestamp_unix_target",
                                  "timestamp_unix_y": "timestamp_unix_seq",
                                  "feature_y": "feature_seq",
                                  "coin_y": "coin_seq",
                                  "coin_x": "coin_target"})

    X_neg = pd.merge(left=neg_df, right=pos_df[["channel_id", "coin", "feature", "timestamp_unix"]], how='left', on=["channel_id"], sort=False)
    X_neg = X_neg[X_neg.timestamp_unix_x > X_neg.timestamp_unix_y]
    X_neg = X_neg.rename(columns={"timestamp_unix_x": "timestamp_unix_target",
                                  "timestamp_unix_y": "timestamp_unix_seq",
                                  "feature": "feature_seq",
                                  "coin_y": "coin_seq",
                                  "coin_x": "coin_target"})

    def udf(df):
        def takeFirst(elem):
            return elem[0]
        # output = []
        feature_seq = []
        coin_seq = []
        X = list(zip(df.timestamp_unix_seq, df.coin_seq, df.feature_seq))
        X.sort(key=takeFirst, reverse=True)
        length = 0
        for x in X:  # set max length to 100
            coin_seq.append(str(x[1]))
            feature_seq.append(str(x[2]))
            length += 1
            if length >= 50:
                break
        return np.array(
            [[df.iloc[0]["channel_id"], df.iloc[0]["coin_target"], df.iloc[0]["timestamp_unix_target"], df.iloc[0]["label"],
              str(length), "\t".join(coin_seq), "\t".join(feature_seq)]])

    X_pos_final = X_pos.groupby(["channel_id", "coin_target", "timestamp_unix_target", "label"]).apply(udf)
    pos_sample_base = pd.DataFrame(np.concatenate(X_pos_final.values, axis=0),
                                   columns=["channel_id", "coin", "timestamp_unix", "label", "length", "coin_seq", "feature_seq"])

    X_neg_final = X_neg.groupby(["channel_id", "coin_target", "timestamp_unix_target", "label"]).apply(udf)
    neg_sample_base = pd.DataFrame(np.concatenate(X_neg_final.values, axis=0),
                                   columns=["channel_id", "coin", "timestamp_unix", "label", "length", "coin_seq", "feature_seq"])


    # channel_coin_sample_base.to_csv("pos_sample_fg.csv", index=False, header=False)
