# coding=utf-8

import tensorflow as tf
from bner.model import train
from absl import flags
from absl import logging
from absl import app


flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .conll files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sentence length after tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "tpu", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Total number of TPU cores to use.")

flags.DEFINE_float("learning_rate", 5e-5, "Initial learning rate for Adam.")

flags.DEFINE_float("warmup_proportion", 0.0, "Proportion of training to perform linear learning rate warmup for.")

flags.DEFINE_float("adam_epsilon", 1e-8, "Epsilon for Adam optimizer.")

flags.DEFINE_float("weight_decay", 0.0, "Weight deay if we apply some.")

flags.DEFINE_integer("batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("epochs", 3, "Total number of training epochs to perform.")

flags.DEFINE_string("gpus", 0, "Comma separated list of gpus devices. If only one, switch to single gpu strategy, if None takes all the gpus available.")



def main(_):
    logging.set_verbosity(logging.INFO)

    train(flags.FLAGS.flag_values_dict())


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("output_dir")
    app.run(main)
