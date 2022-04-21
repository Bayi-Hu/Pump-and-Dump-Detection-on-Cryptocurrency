#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Model(object):

    def __init__(self, tensor_dict, train_config):

        self.label = tensor_dict["label"]
        self.channel_embedding = tensor_dict["channel_embedding"]
        self.coin_embedding = tensor_dict["coin_embedding"]
        self.target_features = tensor_dict["target_features"]

        # .. will add

        # configuration
        self.model_config = {
            "hidden1": 32,
            "hidden2": 32,
            "learning_rate": 0.0005
        }
        self.train_config = train_config
        # is_training, dropout_rate

    def build(self):
        """
        build the architecture for the base DNN model.
        """
        # inp = tf.concat([self.channel_embedding, self.coin_embedding, self.target_features], axis=1)
        # inp = tf.concat([self.channel_embedding, self.coin_embedding], axis=1)
        # self.inp = tf.concat([self.channel_embedding, self.target_features], axis=1)
        # self.inp = self.target_features
        self.inp = self.coin_embedding
        self.build_fcn_net(self.inp)
        self.loss_op()

    def build_fcn_net(self, inp):
        with tf.name_scope("Fully_connected_layer"):

            dnn1 = tf.layers.dense(inp, self.model_config["hidden1"], activation=tf.nn.relu, name='f1')
            # bn1 = tf.layers.batch_normalization(dnn1, training=self.train_config["is_training"], axis=1)
            # dnn1 = tf.layers.dropout(dnn1, self.train_config["dropout_rate"], training=self.train_config["is_training"])

            dnn2 = tf.layers.dense(dnn1, self.model_config["hidden2"], activation=tf.nn.relu, name='f2')
            # bn2 = tf.layers.batch_normalization(dnn2, training=self.train_config["is_training"], axis=1)
            # dnn2 = tf.layers.dropout(dnn2, self.train_config["dropout_rate"], training=self.train_config["is_training"])

            self.logit = tf.squeeze(tf.layers.dense(dnn2, 1, activation=None, name='logit'))

        self.y_hat = tf.sigmoid(self.logit)

        return

    def loss_op(self):
        with tf.name_scope('Metrics'):

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logit))

            # self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=self.label,
            #                                                                      logits=self.dnn3,
            #                                                                      pos_weight=tf.constant(2.0)))
            # Cross-entropy loss and optimizer initialization
            # weight = tf.constant([1.0, 30.0])
            # ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.label)
            # self.loss = ctr_loss
            # add to summary
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.model_config["learning_rate"]).minimize(self.loss)

        self.merged = tf.summary.merge_all()
        return
