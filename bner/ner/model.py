# coding=utf-8
import datetime
import os

import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
import tensorflow_hub as hub
import bert
from bert import tokenization
from bert import optimization
import numpy as np
from bner.ner import dataset
import _pickle as pickle

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"
MAX_SEQ_LENTGH = 0


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def create_model(is_training, input_ids, input_mask, segment_ids, labels, num_labels):
    tags = set()

    if is_training:
        tags.add("train")

    bert_module = hub.Module(BERT_MODEL_HUB, tags=tags, trainable=True)
    bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)
    output_layer = bert_outputs["sequence_output"]

    if is_training:
        output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)

    logits = hidden2tag(output_layer, num_labels)
    logits = tf.reshape(logits, [-1, MAX_SEQ_LENTGH, num_labels])
    mask2len = tf.reduce_sum(input_mask, axis=1)
    loss, trans = crf_loss(logits, labels, input_mask, num_labels, mask2len)
    predict, viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)

    return loss, logits, predict


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_id": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features), batch_size=batch_size,
            num_parallel_calls=8, drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn


def hidden2tag(hiddenlayer, numclass):
    linear = tf.keras.layers.Dense(numclass)

    return linear(hiddenlayer)


def crf_loss(logits, labels, mask, num_labels, mask2len):
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable("transition", shape=[num_labels, num_labels],
                                initializer=tf.contrib.layers.xavier_initializer())

    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels,
                                                                   transition_params=trans,
                                                                   sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)

    return loss, transition


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps, num_warmup_steps, use_tpu):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""
        tf.logging.info("*** Features ***")

        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_id"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, logits, preds = create_model(is_training, input_ids, input_mask, segment_ids,
                                                 label_ids, num_labels)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps,
                                                     num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=total_loss,
                                                          train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, logits, num_labels, mask):
                predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)

                return {
                    "confusion_matrix": _streaming_confusion_matrix(label_ids, predictions,
                                                                    num_labels-1, weights=mask)
                }

            eval_metrics = (metric_fn, [label_ids, logits, num_labels, input_mask])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=total_loss,
                                                          eval_metrics=eval_metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=preds)

        return output_spec

    # Return the actual model function in the closure
    return model_fn


def serving_input_fn():
    label_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENTGH], name='label_id')
    input_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENTGH], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, MAX_SEQ_LENTGH], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENTGH], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_id': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()

    return input_fn


def write_results(output_predict_file, result, batch_tokens, batch_labels, label_list):
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        predictions = []

        for _, pred in enumerate(result):
            predictions.extend(pred)

        for i, prediction in enumerate(predictions):
            token = batch_tokens[i]
            predict = label_list[prediction]
            true_label = label_list[batch_labels[i]]

            if token != "[PAD]" and token != "[CLS]" and true_label != "X":
                if predict == "X" and not token.startswith("##"):
                    predict = "O"
                line = "{}\t{}\t{}\n".format(token, predict, true_label)
                writer.write(line)


def calculate(total_cm, num_class):
    precisions = []
    recalls = []
    f1s = []

    for i in range(num_class):
        rowsum, colsum = np.sum(total_cm[i]), np.sum(total_cm[r][i] for r in range(num_class))
        precision = total_cm[i][i] / float(colsum + 1e-12)
        recall = total_cm[i][i] / float(rowsum + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)


def train(config):
    global MAX_SEQ_LENTGH
    MAX_SEQ_LENTGH = config['max_seq_length']
    tokenizer = create_tokenizer_from_hub_module()
    train_tsv = os.path.join(config['data_dir'], "train.conll")
    test_tsv = os.path.join(config['data_dir'], "test.conll")
    train_tfrecord_file = os.path.join(config['data_dir'], "train.tf_record")
    eval_tfrecord_file = os.path.join(config['data_dir'], "eval.tf_record")
    metadata_file = os.path.join(config['data_dir'], "metadata.pkl")
    output_dir = config['output_dir']

    if not tf.gfile.Exists(train_tfrecord_file) or not tf.gfile.Exists(eval_tfrecord_file) \
            or not tf.gfile.Exists(metadata_file):
        dataset.create_features(MAX_SEQ_LENTGH, tokenizer, train_tsv, test_tsv, train_tfrecord_file,
                                eval_tfrecord_file, metadata_file)

    with tf.gfile.GFile(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    label_list = metadata["labels"]
    train_number_examples = metadata["train_number_examples"]
    predict_eval_number_examples = metadata["eval_number_examples"]
    batch_tokens = metadata["batch_tokens"]
    batch_labels = metadata["batch_labels"]
    tpu_cluster_resolver = None

    if config['use_tpu']:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            config['tpu_name'], zone=config['tpu_zone'], project=config['gcp_project'])

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(cluster=tpu_cluster_resolver, master=config['master'],
                                          model_dir=output_dir,
                                          save_checkpoints_steps=config['save_checkpoints_steps'],
                                          tpu_config=tf.contrib.tpu.TPUConfig(
                                              iterations_per_loop=config['iterations_per_loop'],
                                              num_shards=config['num_tpu_cores'],
                                              per_host_input_for_training=is_per_host))

    num_train_steps = int(train_number_examples / config['train_batch_size'] *
                          config['num_train_epochs'])
    num_warmup_steps = int(num_train_steps * config['warmup_proportion'])

    model_fn = model_fn_builder(num_labels=len(label_list) + 1,
                                learning_rate=config['learning_rate'],
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps, use_tpu=config['use_tpu'])

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(use_tpu=config['use_tpu'], model_fn=model_fn,
                                            config=run_config,
                                            train_batch_size=config['train_batch_size'],
                                            eval_batch_size=config['eval_batch_size'],
                                            predict_batch_size=config['predict_batch_size'])
    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = file_based_input_fn_builder(input_file=train_tfrecord_file,
                                                 seq_length=MAX_SEQ_LENTGH,
                                                 is_training=True,
                                                 drop_remainder=True)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", train_number_examples)
    tf.logging.info("  Batch size = %d", config['train_batch_size'])
    tf.logging.info("  Num steps = %d", num_train_steps)

    current_time = datetime.datetime.now()

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    tf.logging.info("  Training took time = {}".format(datetime.datetime.now() - current_time))

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if config['use_tpu']:
        # Eval will be slightly WRONG on the TPU because it will truncate
        # the last batch.
        assert len(predict_eval_number_examples) % config['eval_batch_size'] == 0
        eval_steps = int(len(predict_eval_number_examples) // config['eval_batch_size'])

    eval_drop_remainder = True if config['use_tpu'] else False
    eval_predict_input_fn = file_based_input_fn_builder(input_file=eval_tfrecord_file,
                                                        seq_length=MAX_SEQ_LENTGH,
                                                        is_training=False,
                                                        drop_remainder=eval_drop_remainder)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", predict_eval_number_examples)
    tf.logging.info("  Batch size = %d", config['eval_batch_size'])

    result = estimator.evaluate(input_fn=eval_predict_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(output_dir, "eval_results.txt")

    with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")

        confusion_matrix = result["confusion_matrix"]
        precision, recall, f1 = calculate(confusion_matrix, len(label_list)-1)

        tf.logging.info("  precision = %s", str(precision))
        tf.logging.info("  recall = %s", str(recall))
        tf.logging.info("  f1 = %s", str(f1))

        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    if config['use_tpu']:
        assert len(predict_eval_number_examples) % config['predict_batch_size'] == 0

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", predict_eval_number_examples)
    tf.logging.info("  Batch size = %d", config['predict_batch_size'])

    result = estimator.predict(input_fn=eval_predict_input_fn)
    output_predict_file = os.path.join(output_dir, "test_results.tsv")
    write_results(output_predict_file, result, batch_tokens, batch_labels, label_list)

    estimator._export_to_tpu = False

    estimator.export_savedmodel(output_dir, serving_input_fn)
