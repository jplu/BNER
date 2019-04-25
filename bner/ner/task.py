# coding=utf-8

import tensorflow as tf
from bner.ner.model import train


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if tf.flags.FLAGS.use_tpu and not tf.flags.FLAGS.tfhub_cache_dir:
        raise ValueError("The option --tfhub_cache_dir must be set if TPU is used")

    train(tf.flags.FLAGS.flag_values_dict())


if __name__ == "__main__":
    tf.flags.mark_flag_as_required("data_dir")
    tf.flags.mark_flag_as_required("output_dir")
    tf.app.run()
