#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Utils.utils import *

class Model(object):

    def __init__(self, tensor_dict, train_config):

        self.label = tensor_dict["label"]
        self.channel_embedding = tensor_dict["channel_embedding"]
        self.coin_embedding = tensor_dict["coin_embedding"]

        # .. will add

        # configuration
        self.model_config = {
            "hidden1": 64,
            "hidden2": 32,
            "learning_rate": 0.001
        }
        self.train_config = train_config
        # is_training, dropout_rate

    def build(self):
        """
        build the architecture for the base DNN model.
        """
        inp = tf.concat([self.channel_embedding, self.coin_embedding], axis=1)
        self.build_fcn_net(inp)
        self.loss_op()

    def build_fcn_net(self, inp):
        with tf.name_scope("Fully_connected_layer"):
            dnn1 = tf.layers.dense(inp, self.model_config["hidden1"], activation=tf.nn.relu, name='f1')
            dnn1 = tf.layers.dropout(dnn1, self.train_config["dropout_rate"], training=self.train_config["is_training"])
            dnn2 = tf.layers.dense(dnn1, self.model_config["hidden2"], activation=tf.nn.relu, name='f2')
            dnn2 = tf.layers.dropout(dnn2, self.train_config["dropout_rate"], training=self.train_config["is_training"])
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')

        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        return

    def loss_op(self):
        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.label)
            self.loss = ctr_loss
            # add to summary
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.model_config["learning_rate"]).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.label), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()
        return
