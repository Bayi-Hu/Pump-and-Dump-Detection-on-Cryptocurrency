#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from NextPumpDetection.Model.model import Model


class ModelSeqPos(Model):
    def __init__(self, tensor_dict, train_config):
        super(ModelSeqPos, self).__init__(tensor_dict, train_config)

        self.length = tensor_dict["length"]
        self.opt_seq_embedding = tensor_dict["opt_seq_embedding"]

        # notice, it should be mask with the length mask ..

        # 设置一个更小的length, 看一下是不是因为选择太多了
        self.length = tf.where(tf.less_equal(self.length, 10), self.length, tf.Variable([10 for i in range(256)]))
        self.sequence_mask = tf.sequence_mask(self.length, maxlen=50, name="sequence_mask")

        dim = self.opt_seq_embedding.get_shape()[-1]
        mask_2d = tf.tile(tf.expand_dims(self.sequence_mask, axis=2), multiples=[1, 1, dim])

        self.masked_opt_seq_embedding = self.opt_seq_embedding * tf.cast(mask_2d, tf.float32) # convert bool to float
        self.pos_score_global = tf.get_variable("pos_score", [1, 50], initializer=tf.zeros_initializer())

        pos_score_global = tf.tile(self.pos_score_global, [256,1])
        paddings = tf.ones_like(pos_score_global) * (-2 ** 32 + 1)

        pos_score = tf.where(tf.logical_not(self.sequence_mask), paddings, pos_score_global)
        self.pos_score = tf.nn.softmax(pos_score)
        self.opt_seq_embedding_mean = tf.reduce_sum(tf.expand_dims(self.pos_score, axis=2) * self.masked_opt_seq_embedding, axis=1)

        # self.opt_seq_embedding_sum = tf.reduce_sum(self.masked_opt_seq_embedding, axis=1)
        # new_length = tf.where(tf.less(self.length,1), self.length+1, self.length)
        # self.opt_seq_embedding_mean = self.opt_seq_embedding_sum/ tf.cast(tf.expand_dims(new_length,axis=1), tf.float32)

    def build(self):
        """
        override the build function
        """
        # inp = tf.concat([self.channel_embedding, self.target_features, self.coin_embedding, self.seq_coin_embedding_sum], axis=1)
        # inp = tf.concat([self.coin_embedding, self.opt_seq_embedding_mean], axis=1)
        # inp = self.coin_embedding
        # self.inp = tf.concat([self.target_features, self.coin_embedding, self.opt_seq_embedding_mean[:,-9:]], axis=1)
        self.inp = tf.concat([self.target_features, self.opt_seq_embedding_mean[:, -9:]], axis=1)
        # inp = self.opt_seq_embedding_mean[:,-9:]
        self.build_fcn_net(self.inp)
        self.loss_op()
