# -*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import pickle as pkl
from datetime import *
import os

# if __name__ == '__main__':
#     # train/validation/test 要根据时间点分，否则会有leakage
#     df = pd.read_csv("pump_sample_raw.csv")
#
#     df.loc[807, "pre_3d_market_cap_btc"] = 140.867
#     df.loc[807, "pre_3d_market_cap_usd"] = 1377987
#     df.loc[1285, "pre_3d_market_cap_usd"] = 1377987
#     df.loc[1285, "pre_3d_market_cap_btc"] = 140.867
#     df.loc[1247, "pre_3d_market_cap_usd"] = 2707118
#     df.loc[1247, "pre_3d_market_cap_btc"] = 225.352
#     df.loc[1248, "pre_3d_market_cap_usd"] = 2625736
#     df.loc[1248, "pre_3d_market_cap_btc"] = 231.999
#
#     for i in [102, 243, 415, 515, 580, 600, 872, 1056, 1094, 1314, 1331]:
#         df.loc[i, "pre_3d_market_cap_btc"] = 279.966
#         df.loc[i, "pre_3d_market_cap_usd"] = 2036283
#     df.loc[480, "pre_3d_market_cap_btc"] = 836.675
#     df.loc[480, "pre_3d_market_cap_usd"] = 8467606

column_list = "label,channel_id,channel_id_new,coin,coin_id,timestamp_unix,length,coin_id_seq,feature_seq,pre_1h_return,pre_1h_price,pre_1h_price_avg,pre_1h_volume,pre_1h_volume_avg,pre_1h_volume_sum,pre_1h_volume_tb,pre_1h_volume_quote,pre_1h_volume_quote_tb,pre_3h_return,pre_3h_price,pre_3h_price_avg,pre_3h_volume,pre_3h_volume_avg,pre_3h_volume_sum,pre_3h_volume_tb,pre_3h_volume_tb_avg,pre_3h_volume_tb_sum,pre_3h_volume_quote,pre_3h_volume_quote_avg,pre_3h_volume_quote_sum,pre_3h_volume_quote_tb,pre_3h_volume_quote_tb_avg,pre_3h_volume_quote_tb_sum,pre_6h_return,pre_6h_price,pre_6h_price_avg,pre_6h_volume,pre_6h_volume_avg,pre_6h_volume_sum,pre_6h_volume_tb,pre_6h_volume_tb_avg,pre_6h_volume_tb_sum,pre_6h_volume_quote,pre_6h_volume_quote_avg,pre_6h_volume_quote_sum,pre_6h_volume_quote_tb,pre_6h_volume_quote_tb_avg,pre_6h_volume_quote_tb_sum,pre_12h_return,pre_12h_price,pre_12h_price_avg,pre_12h_volume,pre_12h_volume_avg,pre_12h_volume_sum,pre_12h_volume_tb,pre_12h_volume_tb_avg,pre_12h_volume_tb_sum,pre_12h_volume_quote,pre_12h_volume_quote_avg,pre_12h_volume_quote_sum,pre_12h_volume_quote_tb,pre_12h_volume_quote_tb_avg,pre_12h_volume_quote_tb_sum,pre_24h_return,pre_24h_price,pre_24h_price_avg,pre_24h_volume,pre_24h_volume_avg,pre_24h_volume_sum,pre_24h_volume_tb,pre_24h_volume_tb_avg,pre_24h_volume_tb_sum,pre_24h_volume_quote,pre_24h_volume_quote_avg,pre_24h_volume_quote_sum,pre_24h_volume_quote_tb,pre_24h_volume_quote_tb_avg,pre_24h_volume_quote_tb_sum,pre_36h_return,pre_36h_price,pre_36h_price_avg,pre_36h_volume,pre_36h_volume_avg,pre_36h_volume_sum,pre_36h_volume_tb,pre_36h_volume_tb_avg,pre_36h_volume_tb_sum,pre_36h_volume_quote,pre_36h_volume_quote_avg,pre_36h_volume_quote_sum,pre_36h_volume_quote_tb,pre_36h_volume_quote_tb_avg,pre_36h_volume_quote_tb_sum,pre_48h_return,pre_48h_price,pre_48h_price_avg,pre_48h_volume,pre_48h_volume_avg,pre_48h_volume_sum,pre_48h_volume_tb,pre_48h_volume_tb_avg,pre_48h_volume_tb_sum,pre_48h_volume_quote,pre_48h_volume_quote_avg,pre_48h_volume_quote_sum,pre_48h_volume_quote_tb,pre_48h_volume_quote_tb_avg,pre_48h_volume_quote_tb_sum,pre_60h_return,pre_60h_price,pre_60h_price_avg,pre_60h_volume,pre_60h_volume_avg,pre_60h_volume_sum,pre_60h_volume_tb,pre_60h_volume_tb_avg,pre_60h_volume_tb_sum,pre_60h_volume_quote,pre_60h_volume_quote_avg,pre_60h_volume_quote_sum,pre_60h_volume_quote_tb,pre_60h_volume_quote_tb_avg,pre_60h_volume_quote_tb_sum,pre_72h_return,pre_72h_price,pre_72h_price_avg,pre_72h_volume,pre_72h_volume_avg,pre_72h_volume_sum,pre_72h_volume_tb,pre_72h_volume_tb_avg,pre_72h_volume_tb_sum,pre_72h_volume_quote,pre_72h_volume_quote_avg,pre_72h_volume_quote_sum,pre_72h_volume_quote_tb,pre_72h_volume_quote_tb_avg,pre_72h_volume_quote_tb_sum,pre_3d_market_cap_usd,pre_3d_market_cap_btc,pre_3d_price_usd,pre_3d_price_btc,pre_3d_volume_usd,pre_3d_volume_btc,pre_3d_twitter_index,pre_3d_reddit_index,pre_3d_alexa_index".split(",")


class FeatGenerator(object):

    def __init__(self, input_file):
        self.input_file = input_file # pos_sample_fg.csv
        self.feat_config = {
            "n_channel": 200,
            "d_channel": 8,
            "n_coin": 378,
            "d_coin": 8,
            "n_seq_feat": 147,
            "n_target_feat": 138,
            "max_length": 50,
            "batch_size": 256,
            "epoch": 3,
        }

    def parse_split(self, line):
        parse_res = tf.string_split([line], delimiter=",")
        values = parse_res.values
        label = values[0]
        channel = values[1]
        channel_id = values[2]
        coin = values[3]
        coin_id = values[4]
        time_stamp = values[5]
        length = values[6]
        coin_id_seq = values[7]
        feature_seq = values[8]
        feature_target = values[9:]
        return label, channel, channel_id, coin, coin_id, time_stamp, length, coin_id_seq, feature_seq, feature_target

    def parse_sequence(self, sequence):
        """
        split the sequence and convert to dense tensor
        """
        split_sequence = tf.string_split(sequence, delimiter="\t")
        split_sequence = tf.sparse_to_dense(sparse_indices=split_sequence.indices,
                                            output_shape=[self.feat_config["batch_size"],
                                                          self.feat_config["max_length"]],
                                            sparse_values=split_sequence.values, default_value="0")

        return split_sequence

    def parse_feature_sequence(self, sequence):
        split_sequence = tf.string_split(sequence, delimiter="\t")
        split_sequence1 = tf.sparse_to_dense(sparse_indices=split_sequence.indices,
                                            output_shape=[self.feat_config["batch_size"],
                                                          self.feat_config["max_length"]],
                                            sparse_values=split_sequence.values,
                                            default_value="".join(["0" for i in range(self.feat_config["n_seq_feat"])]))

        split_sequence1 = tf.reshape(split_sequence1, shape=[self.feat_config["batch_size"]*self.feat_config["max_length"]])

        split_sequence2 = tf.string_split(split_sequence1, delimiter="")
        split_sequence2 = tf.sparse_to_dense(sparse_indices=split_sequence2.indices,
                                             output_shape=[self.feat_config["batch_size"]*self.feat_config["max_length"],
                                                           self.feat_config["n_seq_feat"]],
                                             sparse_values=split_sequence2.values,
                                             default_value="0")

        split_sequence_final = tf.reshape(split_sequence2, shape=[-1, self.feat_config["max_length"], self.feat_config["n_seq_feat"]])

        return split_sequence_final


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

        label, channel, channel_id, coin, coin_id, time_stamp,\
        length, coin_id_seq, feature_seq, feature_target = iterator.get_next()

        seq_coin_id = self.parse_sequence(coin_id_seq)
        seq_feature = self.parse_feature_sequence(feature_seq)

        features = {}
        features["channel"] = channel
        features["channel_id"] = tf.string_to_number(channel_id, out_type=tf.int32)
        features["coin"] = coin
        features["coin_id"] = tf.string_to_number(coin_id, out_type=tf.int32)

        features["timestamp"] = time_stamp
        # features["label"] = tf.one_hot(tf.string_to_number(label, out_type=tf.int32), depth=2)
        features["label"] = tf.string_to_number(label, out_type=tf.float32)
        features["target_features"] = tf.reshape(tf.string_to_number(feature_target, out_type=tf.float32), [-1, self.feat_config["n_target_feat"]])

        features["seq_coin_id"] = tf.string_to_number(seq_coin_id, out_type=tf.int32)
        features["seq_feature"] = tf.string_to_number(seq_feature, out_type=tf.float32)
        features["length"] = tf.string_to_number(length, out_type=tf.int32)

        return features


class TensorGenerator(object):

    def __init__(self):
        pass

    def embedding_layer(self, features, feat_config):
        with tf.name_scope('Embedding_layer'):

            channel_lookup_table = tf.get_variable("channel_embedding_var", [feat_config["n_channel"], feat_config["d_channel"]])
            with open("../../FeatGeneration/wv_embedding.pkl", "rb") as f:
                wv_embedding = pkl.load(f)

            coin_lookup_table = tf.constant(wv_embedding, name="coin_embedding_var", dtype=tf.float32)
            # coin_lookup_table = tf.get_variable("coin_embedding_var", [feat_config["n_coin"], feat_config["d_coin"]])

            # add to summary
            # tf.summary.histogram('channel_lookup_table', channel_lookup_table)
            # tf.summary.histogram('coin_lookup_table', coin_lookup_table)

            channel_embedding = tf.nn.embedding_lookup(channel_lookup_table, features["channel_id"])
            coin_embedding = tf.nn.embedding_lookup(coin_lookup_table, features["coin_id"])

            # # coin sequence
            # seq_coin_embedding = tf.nn.embedding_lookup(coin_lookup_table,
            #                                             tf.string_to_hash_bucket_fast(features["seq_coin"],
            #                                                                           feat_config["n_coin"]))

            seq_coin_embedding = tf.nn.embedding_lookup(coin_lookup_table, features["seq_coin_id"])

            # concatenate the tensors
            tensor_dict = {}

            tensor_dict["label"] = features["label"]
            tensor_dict["length"] = features["length"]
            tensor_dict["channel_embedding"] = channel_embedding
            tensor_dict["coin_embedding"] = coin_embedding
            tensor_dict["target_features"] = features["target_features"]
            # tensor_dict["opt_seq_coin_embedding"] = seq_coin_embedding
            tensor_dict["opt_seq_embedding"] = features["seq_feature"]
            tensor_dict["opt_seq_coin_embedding"] = seq_coin_embedding
            # tensor_dict["opt_seq_embedding"] = seq_coin_embedding
            # tensor_dict["opt_seq_embedding"] = tf.concat([seq_coin_embedding, features["seq_feature"]], 2)

        return tensor_dict

if __name__ == '__main__':

    fg = FeatGenerator("feature/test_sample.csv")
    features = fg.feature_generation()

    tg = TensorGenerator()
    tensor_dict = tg.embedding_layer(features, fg.feat_config)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tensor_dict["opt_seq_embedding"])
    sess.run(tensor_dict["label"])
    sess.run(tensor_dict["target_features"])

