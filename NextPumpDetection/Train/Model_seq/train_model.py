#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from NextPumpDetection.FeatGeneration.data_loader import FeatGenerator, TensorGenerator
from NextPumpDetection.Model.model import Model

if __name__ == '__main__':

    train_file = "../../FeatGeneration/test_sample.csv"

    train_fg = FeatGenerator(train_file)
    train_features = train_fg.feature_generation()
    tg = TensorGenerator()
    train_tensor_dict = tg.embedding_layer(train_features, train_fg.feat_config)

    # test_fg = FeatGenerator(test_file)
    # test_features = test_fg.feature_generation()
    # test_tensor_dict = tg.embedding_layer(test_features, test_fg.feat_config)
    model = Model(train_tensor_dict, train_config={"is_training": True, "dropout_rate": 0})
    model.build()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iter = 0
        while True:
        # for i in range(100):
            try:
                _, loss, acc, l, y_ = sess.run([model.optimizer, model.loss, model.accuracy, model.label, model.y_hat])

                #
                if iter % 10 == 0:
                    print("iter=%d, loss=%f, acc=%f" %(iter, loss, acc))

                iter += 1
            except Exception as e:
                print(e)
                break


