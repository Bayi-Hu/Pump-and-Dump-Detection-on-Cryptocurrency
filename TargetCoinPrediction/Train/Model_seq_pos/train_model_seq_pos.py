#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from TargetCoinPrediction.FeatGeneration.data_loader import FeatGenerator, TensorGenerator
from TargetCoinPrediction.Model.model_seq_pos import ModelSeqPos
from sklearn import metrics
import numpy as np
import os

if __name__ == '__main__':

    train_file = "../../FeatGeneration/feature/train_sample.csv"; sample_num = 132314

    train_fg = FeatGenerator(train_file)
    train_fg.feat_config["epoch"] = 30
    train_features = train_fg.feature_generation()
    tg = TensorGenerator()
    train_tensor_dict = tg.embedding_layer(train_features, train_fg.feat_config)

    model = ModelSeqPos(train_tensor_dict, train_config={"is_training": True, "dropout_rate": 0})
    model.build()

    checkpoint_dir = "./save_log"
    saver = tf.train.Saver(max_to_keep = 50)
    save_iter = int(sample_num / train_fg.feat_config["batch_size"])

    pred_probas = []
    labels = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iter = 0
        while True:
        # for i in range(100):
            try:
                _, loss, l, y_ = sess.run([model.optimizer, model.loss, model.label, model.y_hat])

                pred_probas += list(y_)
                labels += list(l)

                if iter % 10 == 0:
                    fpr, tpr, thresholds = metrics.roc_curve(labels, pred_probas, pos_label=1)
                    auc_value = metrics.auc(fpr, tpr)
                    print("iter=%d, loss=%f, auc=%f" %(iter, loss, auc_value))

                iter += 1

                if iter % save_iter == 0 and iter > 0:
                    saver.save(sess, os.path.join(checkpoint_dir, "model_seq_pos" + str(round(iter / save_iter))))

            except Exception as e:
                print(e)
                break
