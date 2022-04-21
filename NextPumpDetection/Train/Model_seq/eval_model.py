#-*- coding:utf-8 -*-
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from NextPumpDetection.FeatGeneration.data_loader import FeatGenerator, TensorGenerator
from NextPumpDetection.Model.model import Model
from sklearn import metrics
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

if __name__ == '__main__':

    test_file = "../../FeatGeneration/feature/test_nolog_norm_sample.csv"

    test_fg = FeatGenerator(test_file)
    test_fg.feat_config["epoch"] = 1
    test_features = test_fg.feature_generation()
    tg = TensorGenerator()
    test_tensor_dict = tg.embedding_layer(test_features, test_fg.feat_config)

    # test_fg = FeatGenerator(test_file)
    # test_features = test_fg.feature_generation()
    # test_tensor_dict = tg.embedding_layer(test_features, test_fg.feat_config)
    model = Model(test_tensor_dict, train_config={"is_training": False, "dropout_rate": 0})
    model.build()

    auc_value_list = []
    for epoch in range(4,31):
        ckpt = "./save_log/model_" + str(epoch)
        saver = tf.train.Saver()

        pre_probas = []
        labels = []
        coin_ids = []
        channel_ids = []
        timestamps = []

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            try:
                saver.restore(sess, ckpt)
            except:
                break
            iter = 0
            while True:
            # for i in range(10):

                try:
                    channel_id, coin_id, timestamp, y_, label, loss = \
                                                sess.run([test_features["channel"],
                                                test_features["coin"],
                                                test_features["timestamp"],
                                                model.y_hat,
                                                model.label,
                                                model.loss])

                    pre_probas += list(y_)
                    labels += list(label)
                    timestamps += list(timestamp)
                    channel_ids += list(channel_id)
                    coin_ids += list(coin_id)

                    # if iter % 10 == 0:
                    #     print("iter=%d, loss=%f, acc=%f" %(iter, loss))
                    iter += 1
                except Exception as e:
                    # print(e)
                    break

        fpr, tpr, thresholds_roc = metrics.roc_curve(y_true=labels, y_score=pre_probas, pos_label=1)
        auc_value = metrics.auc(fpr, tpr)
        auc_value_list.append(auc_value)

        print("epoch=%d, auc=%f" %(epoch,auc_value))

    print("pause")


test_df = pd.DataFrame(
            data={"channel_id": channel_ids,
                  "coin": coin_ids,
                  "timestamp": timestamps,
                  "label": labels,
                  "y_pred_proba": pre_probas},
            dtype = int
)

def hitrate(k):

    def udf(df):
        def takeFirst(elem):
            return elem[0]
        X = list(zip(df.y_pred_proba, df.label))
        X.sort(key=takeFirst, reverse=True)
        #     permute = np.random.permutation(len(X))
        labels = []
        for i in range(k):
            #         x = X[permute[i]]
            x = X[i]
            labels.append(x[1])

        return np.array(
            [[df.iloc[0]["channel_id"], df.iloc[0]["timestamp"], np.sum(labels)]])

    x_test = test_df[["channel_id", "timestamp", "label", "y_pred_proba"]].groupby(["channel_id", "timestamp"]).apply(
        udf)
    test_hitrate = pd.DataFrame(np.concatenate(x_test.values, axis=0),
                                columns=["channel_id", "timestamp", "label_num"])

    test_hitrate.label_num.value_counts()
    return test_hitrate.label_num.mean()


threshold = 0.1
y_pred = np.zeros_like(pre_probas)
y_pred[np.where(np.array(pre_probas) >= threshold)[0]] = 1
print(np.sum(y_pred))
print(classification_report(labels, y_pred))

precision, recall, thresholds_pr = precision_recall_curve(labels, pre_probas)
plt.figure("P-R Curve")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot(recall, precision)

#
fpr, tpr, thresholds = metrics.roc_curve(labels, pre_probas, pos_label=1)
metrics.auc(fpr, tpr)
plt.figure("ROC Curve")
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.plot(fpr, tpr)

# hitrate calculation

