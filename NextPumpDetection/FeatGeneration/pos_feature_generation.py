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
    df = pd.read_csv("pump_sample_raw.csv")

    df.loc[807, "pre_3d_market_cap_btc"] = 140.867
    df.loc[807, "pre_3d_market_cap_usd"] = 1377987
    df.loc[1285, "pre_3d_market_cap_usd"] = 1377987
    df.loc[1285, "pre_3d_market_cap_btc"] = 140.867
    df.loc[1247, "pre_3d_market_cap_usd"] = 2707118
    df.loc[1247, "pre_3d_market_cap_btc"] = 225.352
    df.loc[1248, "pre_3d_market_cap_usd"] = 2625736
    df.loc[1248, "pre_3d_market_cap_btc"] = 231.999

    for i in [102, 243, 415, 515, 580, 600, 872, 1056, 1094, 1314, 1331]:
        df.loc[i, "pre_3d_market_cap_btc"] = 279.966
        df.loc[i, "pre_3d_market_cap_usd"] = 2036283
    df.loc[480, "pre_3d_market_cap_btc"] = 836.675
    df.loc[480, "pre_3d_market_cap_usd"] = 8467606

    df = df[df["exchange"] == "binance"]
    df = df[df["pair"] == "BTC"]

    #其余的使用平均值填充
    for column in ['pre_3d_market_cap_usd','pre_3d_market_cap_btc', 'pre_3d_price_usd', 'pre_3d_price_btc','pre_3d_volume_usd', 'pre_3d_volume_btc', 'pre_3d_twitter_index','pre_3d_reddit_index', 'pre_3d_alexa_index']:
        mean_val = df[column].mean()
        df[column].fillna(mean_val, inplace=True)

    price_columns = []
    return_columns = []
    volume_columns = []

    for column in df.columns:
        if "price" in column:
            price_columns.append(column)
        if "return" in column:
            return_columns.append(column)
        if "volume" in column:
            volume_columns.append(column)

    other_columns = ['pre_3d_alexa_index', 'pre_3d_market_cap_btc', 'pre_3d_market_cap_usd', 'pre_3d_reddit_index', 'pre_3d_twitter_index']

    for column in price_columns:
        df[column] = df[column] * (10**5)

    for column in volume_columns + other_columns:
        df[column] = np.log2(df[column]+0.1)

    # sequence 化
    for idx, row in df.iterrows():
        feature = []
        for column in price_columns + return_columns + volume_columns:
            feature.append(row[column])

        feature_str = "".join(map(lambda x: str(x), feature))
        df.loc[idx, "feature"] = feature_str
        


    pre_feature_columns = []

    X = pd.merge(left=df, right=df[["channel_id", "coin", "feature", "timestamp"]], how='left', on=["channel_id"],
                 sort=False)

    X = X[X.timestamp_x > X.timestamp_y]


    def udf(df):
        def takeFirst(elem):
            return elem[0]
        # output = []
        feature_seq = []
        coin_seq = []
        X = list(zip(df.timestamp_y, df.coin_y, df.feature_y))
        X.sort(key=takeFirst, reverse=True)
        length = 0
        for x in X:  # set max length to 100
            coin_seq.append(str(x[1]))
            feature_seq.append(str(x[2]))
            length += 1
            if length >= 50:
                break
        return np.array(
            [[df.iloc[0]["channel_id"], df.iloc[0]["coin_x"], df.iloc[0]["timestamp_x"], df.iloc[0]["session_id"],
              str(length), "\t".join(coin_seq), "\t".join(feature_seq)]])

    X_ = X.groupby(["channel_id", "coin_x", "timestamp_x", "session_id"]).apply(udf)
    channel_coin_sample_base = pd.DataFrame(np.concatenate(X_.values, axis=0), columns=["channel_id", "coin", "timestamp", "session_id", "length", "coin_seq", "feature_seq"])
    channel_coin_sample_base.to_csv("pos_sample_fg.csv", index=False, header=False)



class FeatGenerator(object):

    def __init__(self, input_file):
        self.input_file = input_file # pos_sample_fg.csv
        self.feat_config = {
            "n_channel": 100,
            "d_channel": 16,
            "n_coin": 1000,
            "n_feat": 142,
            "max_length": 50,
            "batch_size": 128,
            "epoch": 3
        }

    def parse_split(line):
        parse_res = tf.string_split([line], delimiter=",")
        values = parse_res.values
        channel = values[0]
        coin = values[1]
        length = values[4]
        coin_seq = values[5]
        feature_seq = values[6]

        return channel, coin, coin_seq, feature_seq, length


    def parse_sequence(self, sequence):
        """
        split the sequence and convert to dense tensor
        """
        split_sequence = tf.string_split(sequence, delimiter="\t")
        split_sequence = tf.string_split(sequence, delimiter="")
        split_sequence = tf.sparse_to_dense(sparse_indices=split_sequence.indices,
                                            output_shape=[self.feat_config["batch_size"],
                                                          self.feat_config["max_length"]],
                                            sparse_values=split_sequence.values, default_value="0")

        return split_sequence

    def parse_feature_sequence(self, sequence):
        split_sequence = tf.string_split(feature_seq, delimiter="\t")
        split_sequence = tf.sparse_to_dense(sparse_indices=split_sequence.indices,
                                            output_shape=[10, 50],
                                            sparse_values=split_sequence.values,
                                            default_value="".join(["0" for i in range(142)]))
        split_sequence1 = tf.reshape(split_sequence, shape=[10 * 50])
        split_sequence2 = tf.string_split(split_sequence1, delimiter="")
        split_sequence3 = tf.sparse_to_dense(sparse_indices=split_sequence2.indices,
                                             output_shape=[500, 142],
                                             sparse_values=split_sequence2.values, default_value="0")

        split_sequence_final = tf.reshape(split_sequence3, shape=[10, 50, 143])

        return split_sequence3


    def feature_generation(self):
        """
        Args:
            input_file: a .txt file that includes the training or testing sample
        Returns:
            feature tensor used for training or testing
        """
        dataset = tf.data.TextLineDataset(self.input_file)
        dataset = dataset.map(self.parse_split, num_parallel_calls=2)
        dataset = dataset.shuffle(3).repeat(self.feat_config["epoch"]).batch(self.feat_config["batch_size"])
        iterator = dataset.make_one_shot_iterator()

        channel, coin, coin_seq, feature_seq, length = iterator.get_next()
        seq_coin = self.parse_sequence(coin_seq)
        seq_feature = self.parse_feature_sequence(feature_seq)

        features = {}
        features["channel"] = channel
        features["coin"] = coin

        features["seq_coin"] = seq_coin
        features["seq_feature"] = seq_feature
        features["length"] = tf.string_to_number(length, out_type=tf.int32)

        return features




# class TensorGenerator(object):
#
#     def __init__(self):
#         pass
#
#     def embedding_layer(self, features, feat_config):
#
#         with tf.name_scope('Embedding_layer'):
#
#             uid_lookup_table = tf.get_variable("uid_embedding_var", [feat_config["n_uid"], feat_config["d_uid"]])
#             iid_lookup_table = tf.get_variable("iid_embedding_var", [feat_config["n_iid"], feat_config["d_iid"]])
#             cat_lookup_table = tf.get_variable("cat_embedding_var", [feat_config["n_cid"], feat_config["d_cid"]])
#
#             # add to summary
#             # tf.summary.histogram('uid_lookup_table', uid_lookup_table)
#             # tf.summary.histogram('iid_lookup_table', iid_lookup_table)
#             # tf.summary.histogram('cat_lookup_table', cat_lookup_table)
#
#             uid_embedding = tf.nn.embedding_lookup(uid_lookup_table,
#                                                    tf.string_to_hash_bucket_fast(features["user_id"],
#                                                                                  feat_config["n_uid"]))
#
#             iid_embedding = tf.nn.embedding_lookup(iid_lookup_table,
#                                                    tf.string_to_hash_bucket_fast(features["item_id"],
#                                                                                  feat_config["n_iid"]))
#
#             cat_embedding = tf.nn.embedding_lookup(cat_lookup_table,
#                                                    tf.string_to_hash_bucket_fast(features["category"],
#                                                                                  feat_config["n_cid"]))
#
#             # item sequence
#             seq_iid_embedding = tf.nn.embedding_lookup(iid_lookup_table,
#                                                        tf.string_to_hash_bucket_fast(features["seq_item_id"],
#                                                                                      feat_config["n_iid"]))
#
#             seq_cat_embedding = tf.nn.embedding_lookup(cat_lookup_table,
#                                                        tf.string_to_hash_bucket_fast(features["seq_category"],
#                                                                                      feat_config["n_cid"]))
#
#             # concatenate the tensors
#             tensor_dict = {}
#             tensor_dict["user_embedding"] = uid_embedding
#             tensor_dict["item_embedding"] = tf.concat([iid_embedding, cat_embedding], 1)
#             tensor_dict["opt_seq_embedding"] = tf.concat([seq_iid_embedding, seq_cat_embedding], 2)
#             tensor_dict["length"] = features["length"]
#             tensor_dict["label"] = features["label"]
#
#         return tensor_dict