#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Model.model import Model


class ModelSeq(Model):
    def __init__(self, tensor_dict, train_config):
        super(ModelSeq, self).__init__(tensor_dict, train_config)

        self.length = tensor_dict["length"]
        self.opt_seq_embedding = tensor_dict["opt_seq_embedding"]

        # notice, it should be mask with the length mask ..
        self.sequence_mask = tf.sequence_mask(self.length, maxlen=50, name="sequence_mask")
        dim = self.opt_seq_embedding.get_shape()[-1]
        mask_2d = tf.tile(tf.expand_dims(self.sequence_mask, axis=2), multiples=[1, 1, dim])
        self.masked_opt_seq_embedding = self.opt_seq_embedding * tf.cast(mask_2d, tf.float32) # convert bool to float
        self.seq_coin_embedding_sum = tf.reduce_sum(self.masked_opt_seq_embedding, axis=1)

    def build(self):
        """
        override the build function
        """
        inp = tf.concat([self.channel_embedding, self.coin_embedding, self.seq_coin_embedding_sum], axis=1)
        self.build_fcn_net(inp)
        self.loss_op()
