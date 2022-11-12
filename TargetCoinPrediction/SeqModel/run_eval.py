#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from run_train import *

def del_flags(FLAGS, keys_list):
    for keys in keys_list:
        FLAGS.__delattr__(keys)

    return

if __name__ == '__main__':
    del_flags(FLAGS, ["do_train", "do_eval", "init_checkpoint"])
    flags.DEFINE_bool("do_train", False, "")
    flags.DEFINE_bool("do_eval", True, "")
    flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint.")
    tf.app.run()


