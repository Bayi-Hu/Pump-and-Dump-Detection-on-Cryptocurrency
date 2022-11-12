import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from data_loader import FeatGenerator
from modeling import DNN, SNN, SNNTA
from sklearn import metrics
import pandas as pd
import numpy as np
import time
import optimization
import collections
import re
import os

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "train_input_file", "../FeatGeneration/feature/train_sample.csv",
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "test_input_file", "../FeatGeneration/feature/test_sample.csv",
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "checkpointDir", "ckpt_dir",
    "The output directory where the model checkpoints will be written.")

## hyper-parameters
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_integer("max_seq_length", 10, "")
flags.DEFINE_bool("do_train", True, "")
flags.DEFINE_bool("do_eval", False, "")
flags.DEFINE_integer("batch_size", 256, "")
flags.DEFINE_integer("epoch", 30, "")
flags.DEFINE_float("learning_rate", 5e-4, "")
flags.DEFINE_integer("num_train_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer("num_warmup_steps", 100, "Number of warmup steps.")
flags.DEFINE_integer("save_checkpoints_steps", 8000, "")
flags.DEFINE_integer("iterations_per_loop", 5000, "How many steps to make in each estimator call.")
flags.DEFINE_integer("max_eval_steps", 1000, "Maximum number of eval steps.")
flags.DEFINE_float("dropout_rate", 0.2, "The dropout rate in training." )
flags.DEFINE_string("model", "snn", "model type {dnn, snn}")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string("data_dir", './data/', "data dir.")

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


# def model_fn(features, mode, init_checkpoint, learning_rate,
#              num_train_steps, num_warmup_steps, use_tpu):
#     """The `model_fn` for TPUEstimator."""
#
#     tf.logging.info("*** Features ***")
#     for name in sorted(features.keys()):
#         tf.logging.info("name = %s, shape = %s" % (name,
#                                                    features[name].shape))
#
#     is_training = (mode == tf.estimator.ModeKeys.TRAIN)
#
#     if FLAGS.model == "snn":
#         model = SNN(features, FLAGS.telegram)
#
#     elif FLAGS.model == "dnn":
#         model = DNN(features, FLAGS.telegram)
#
#     else:
#         raise ValueError("Should select correct model name:", FLAGS.model)
#
#     tvars = tf.trainable_variables()
#     initialized_variable_names = {}
#
#     if init_checkpoint:
#         (assignment_map, initialized_variable_names
#          ) = get_assignment_map_from_checkpoint(
#             tvars, init_checkpoint)
#         if use_tpu:
#
#             def tpu_scaffold():
#                 tf.train.init_from_checkpoint(init_checkpoint,
#                                               assignment_map)
#                 return tf.train.Scaffold()
#
#             scaffold_fn = tpu_scaffold
#         else:
#             tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
#
#     tf.logging.info("**** Trainable Variables ****")
#     for var in tvars:
#         init_string = ""
#         if var.name in initialized_variable_names:
#             init_string = ", *INIT_FROM_CKPT*"
#         tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
#                         init_string)
#
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         train_op = optimization.create_optimizer(model.opt_loss, learning_rate,
#                                                  num_train_steps,
#                                                  num_warmup_steps, use_tpu)
#
#         return model, train_op
#
#     elif mode == tf.estimator.ModeKeys.EVAL:
#
#         return model
#     else:
#         raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))


# def main(_):
#
#     if FLAGS.do_train:
#         mode = tf.estimator.ModeKeys.TRAIN
#
#         input_files = "../Data/data/" + FLAGS.dataset + ".train.tfrecord" + "." + FLAGS.bizdate
#         # load Data
#         features = input_fn(input_files, is_training=True)
#
#     elif FLAGS.do_eval:
#         mode = tf.estimator.ModeKeys.EVAL
#         input_files = "../Data/data/" + FLAGS.dataset + ".test.tfrecord" + "." + FLAGS.bizdate
#         features = input_fn(input_files, is_training=False)
#
#     else:
#         raise ValueError("Only TRAIN and EVAL modes are supported.")
#
#     # modeling
#     tf.gfile.MakeDirs(FLAGS.checkpointDir)
#
#     if FLAGS.do_train:
#         model, train_op = model_fn(features, mode, FLAGS.init_checkpoint, FLAGS.learning_rate,
#                                    FLAGS.num_train_steps, FLAGS.num_warmup_steps, False)
#         # saver define
#         tvars = tf.trainable_variables()
#         saver = tf.train.Saver(max_to_keep=30, var_list=tvars)
#
#         # start session
#         config = tf.ConfigProto(allow_soft_placement=True)
#         config.gpu_options.allow_growth = True
#
#         with tf.Session(config=config) as sess:
#             sess.run(tf.global_variables_initializer())
#             total_loss = []
#             losses = []
#             iter = 0
#             start = time.time()
#             while True:
#                 try:
#                     _, loss = sess.run([train_op, model.real_loss])
#                     losses.append(loss)
#
#                     if iter % 50 == 0:
#                         end = time.time()
#                         loss = np.mean(np.concatenate(losses))
#                         total_loss.append(loss)
#                         print("iter=%d, MAE_loss=%f, time=%.2fs" % (iter, loss, end - start))
#                         losses = []
#                         start = time.time()
#
#                     if iter % FLAGS.save_checkpoints_steps == 0 and iter >= 10000:
#                         saver.save(sess, os.path.join(FLAGS.checkpointDir, FLAGS.model +"_" + str(round(iter))))
#                     iter += 1
#
#                 except Exception as e:
#                     # print("Out of Sequence, end of training...")
#                     print(e)
#                     # save model
#                     saver.save(sess, os.path.join(FLAGS.checkpointDir, FLAGS.model +"_" + str(round(iter))))
#                     break
#
#     elif FLAGS.do_eval:
#         # must have checkpoint
#         if FLAGS.init_checkpoint==None:
#             raise ValueError("Must need a checkpoint for evaluation")
#
#         model = model_fn(features, mode, FLAGS.init_checkpoint, FLAGS.learning_rate,
#                                           FLAGS.num_train_steps, FLAGS.num_warmup_steps, False)
#
#         # start session
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#
#             losses = []
#             iter = 0
#             start = time.time()
#
#             while True:
#                 try:
#                     loss = sess.run(model.real_loss)
#                     losses.append(loss)
#
#                     if iter % 500 == 0:
#                         end = time.time()
#                         print("iter=%d, time=%.2fs" % (iter, end - start))
#                         start = time.time()
#
#                     iter += 1
#
#                 except Exception as e:
#                     print("Out of Sequence")
#                     # save model
#                     # saver.save(sess, os.path.join(FLAGS.checkpointDir, FLAGS.model + "_" + str(iter)))
#                     break
#             losses = np.concatenate(losses)
#             final_loss = np.mean(losses)
#             eval_sample_num = len(losses)
#
#             print("========Evaluation Results==========")
#             print("sample_num=%d, MAE_loss=%f" % (eval_sample_num, final_loss))
#
#     else:
#         raise ValueError("Only TRAIN and EVAL modes are supported.")
#     return

def HitRatio_calculation(channel_ids, coin_ids, timestamps, labels, pre_probas):
    test_df = pd.DataFrame(
        data={"channel_id": channel_ids,
              "coin": coin_ids,
              "timestamp": timestamps,
              "label": labels,
              "y_pred_proba": pre_probas},
        dtype=int
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

        x_test = test_df[["channel_id", "timestamp", "label", "y_pred_proba"]].\
                        groupby(["channel_id", "timestamp"]).apply(udf)


        test_hitrate = pd.DataFrame(np.concatenate(x_test.values, axis=0),
                                    columns=["channel_id", "timestamp", "label_num"])

        test_hitrate.label_num = test_hitrate.label_num.astype(int)
        # test_hitrate.label_num.value_counts()
        return test_hitrate.label_num.mean()

    HR1 = hitrate(1)
    HR3 = hitrate(3)
    HR5 = hitrate(5)
    HR10 = hitrate(10)
    HR20 = hitrate(20)
    HR50 = hitrate(50)

    return HR1, HR3, HR5, HR10, HR20, HR50


def model_fn(tensor_dict):

    model_config = {"is_training": FLAGS.do_train,
                    "dropout_rate": FLAGS.dropout_rate if FLAGS.do_train else 0.0,
                    "max_seq_length": FLAGS.max_seq_length,
                    "learning_rate": FLAGS.learning_rate}

    if FLAGS.model == "snn":
        model = SNN(tensor_dict, model_config)

    elif FLAGS.model == "snnta":
        model = SNNTA(tensor_dict, model_config)

    elif FLAGS.model == "dnn":
        model = DNN(tensor_dict, model_config)

    else:
        raise ValueError("Should select correct model name:", FLAGS.model)

    model.build()

    return model

def main(_):

    sample_num = 132314
    # feature generation
    if FLAGS.do_train:
        feat_generator = FeatGenerator(FLAGS.train_input_file, FLAGS.epoch, FLAGS.batch_size)

    elif FLAGS.do_eval:
        feat_generator = FeatGenerator(FLAGS.test_input_file, 1, FLAGS.batch_size)

    else:
        raise ValueError("Only TRAIN and EVAL mode supported.")

    # build model
    model = model_fn(feat_generator.tensor_dict)

    # define savor
    saver = tf.train.Saver(max_to_keep=50)
    save_iter = int(sample_num / FLAGS.batch_size)

    # TRAINING
    if FLAGS.do_train:
        # intermediate results
        pred_probas = []
        labels = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iter = 0
            start = time.time()
            while True:
                try:
                    _, loss, l, y_ = sess.run([model.optimizer, model.loss, model.label, model.y_hat])

                    pred_probas += list(y_)
                    labels += list(l)

                    if iter % 50 == 0 and iter > 0:
                        end = time.time()
                        fpr, tpr, thresholds = metrics.roc_curve(labels, pred_probas, pos_label=1)
                        auc_value = metrics.auc(fpr, tpr)
                        print("iter=%d, loss=%f, auc=%f, time=%.2fs" % (iter, loss, auc_value, end - start))
                        # clear
                        pred_probas = []
                        labels = []
                        start = time.time()

                    iter += 1

                    if iter % save_iter == 0 and iter > 4:
                        saver.save(sess, os.path.join(FLAGS.checkpointDir, FLAGS.model + str(round(iter / save_iter))))

                except Exception as e:
                    print(e)
                    break

    # EVALUATION
    elif FLAGS.do_eval:

        auc_value_list = []
        for epoch in range(5, 31):

            ckpt = os.path.join(FLAGS.checkpointDir, FLAGS.model + str(epoch))
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
                except Exception as e:
                    print(e)
                    break

                iter = 0
                while True:
                    # for i in range(10):

                    try:
                        channel_id, coin_id, timestamp, y_, label, loss = \
                            sess.run([feat_generator.features["channel"],
                                      feat_generator.features["coin"],
                                      feat_generator.features["timestamp"],
                                      model.y_hat,
                                      model.label,
                                      model.loss])

                        pre_probas += list(y_)
                        labels += list(label)
                        timestamps += list(timestamp)
                        channel_ids += list(channel_id)
                        coin_ids += list(coin_id)

                        iter += 1
                    except Exception as e:
                        # print(e)
                        break

            fpr, tpr, thresholds_roc = metrics.roc_curve(y_true=labels, y_score=pre_probas, pos_label=1)
            auc_value = metrics.auc(fpr, tpr)
            auc_value_list.append(auc_value)

            print("=====================================")
            print("epoch=%d, auc=%f" % (epoch, auc_value))
            HR1, HR3, HR5, HR10, HR20, HR30 = HitRatio_calculation(channel_ids, coin_ids, timestamps, labels, pre_probas)
            print("HitRate@1=%.4f" % HR1)
            print("HitRate@3=%.4f" % HR3)
            print("HitRate@5=%.4f" % HR5)
            print("HitRate@10=%.4f" % HR10)
            print("HitRate@20=%.4f" % HR20)
            print("HitRate@30=%.4f" % HR30)


if __name__ == '__main__':
    tf.app.run()


# threshold = 0.1
# y_pred = np.zeros_like(pre_probas)
# y_pred[np.where(np.array(pre_probas) >= threshold)[0]] = 1
# print(np.sum(y_pred))
# print(classification_report(labels, y_pred))
#
# precision, recall, thresholds_pr = precision_recall_curve(labels, pre_probas)
# plt.figure("P-R Curve")
# plt.title("Precision-Recall Curve")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.plot(recall, precision)
#
# #
# fpr, tpr, thresholds = metrics.roc_curve(labels, pre_probas, pos_label=1)
# metrics.auc(fpr, tpr)
# plt.figure("ROC Curve")
# plt.xlabel("fpr")
# plt.ylabel("tpr")
# plt.plot(fpr, tpr)
