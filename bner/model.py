# coding=utf-8
import datetime
import os

import tensorflow as tf
import numpy as np
from seqeval import metrics
from bner import dataset
import _pickle as pickle
from absl import logging
from transformers import BertConfig, BertTokenizer, TFBertForTokenClassification
from bner import optimization
from fastprogress import master_bar, progress_bar


MAX_SEQ_LENTGH = 0

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})


def file_based_dataset_builder(input_file, is_training, drop_remainder, batch_size):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([MAX_SEQ_LENTGH], tf.int64),
        "input_mask": tf.io.FixedLenFeature([MAX_SEQ_LENTGH], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([MAX_SEQ_LENTGH], tf.int64),
        "label_ids": tf.io.FixedLenFeature([MAX_SEQ_LENTGH], tf.int64),
    }

    def _decode_record(record):
        example = tf.io.parse_single_example(record, name_to_features)
        features = {}
        features['input_ids'] = example['input_ids']
        features['input_mask'] = example['input_mask']
        features['segment_ids'] = example['segment_ids']

        return features, example['label_ids']

    d = tf.data.TFRecordDataset(input_file)

    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=8192)

    d = d.map(_decode_record, num_parallel_calls=4)
    d = d.batch(batch_size, drop_remainder)
    d = d.prefetch(buffer_size=batch_size)

    return d


def write_results(output_predict_file, preds, batch_tokens, gold_labels):
    with tf.io.gfile.GFile(output_predict_file, "w") as writer:
        logging.info("***** Predict results *****")

        for tokens, predicts, true_labels in zip(batch_tokens, preds, gold_labels):
            for token, predict, true_label in zip(tokens, predicts, true_labels):
                writer.write("{}\t{}\t{}\n".format(token, predict, true_label))


def evaluate(model, labels_list, eval_dataset):
    preds = None

    for features, labels in eval_dataset:
        inputs = {'attention_mask': features['input_mask'], 'token_type_ids': features['segment_ids'], 'training': False}
        logits = model(features['input_ids'], **inputs)[0]

        if preds is None:
            preds = logits.numpy()
            label_ids = labels.numpy()
        else:
            preds = np.append(preds, logits.numpy(), axis=0)
            label_ids = np.append(label_ids, labels.numpy(), axis=0)

    preds = np.argmax(preds, axis=2)
    y_pred = [[] for _ in range(label_ids.shape[0])]
    y_true = [[] for _ in range(label_ids.shape[0])]

    for i in range(label_ids.shape[0]):
        for j in range(label_ids.shape[1]):
            if label_ids[i, j] != 0:
                y_pred[i].append(labels_list[preds[i, j] - 1])
                y_true[i].append(labels_list[label_ids[i, j] - 1])

    return y_true, y_pred


def train(args):
    global MAX_SEQ_LENTGH

    MAX_SEQ_LENTGH = args['max_seq_length']
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
    train_tsv = os.path.join(args['data_dir'], "train.conll")
    test_tsv = os.path.join(args['data_dir'], "test.conll")
    train_tfrecord_file = os.path.join(args['data_dir'], "train.tf_record")
    eval_tfrecord_file = os.path.join(args['data_dir'], "eval.tf_record")
    metadata_file = os.path.join(args['data_dir'], "metadata.pkl")
    output_dir = args['output_dir']

    if not tf.io.gfile.exists(train_tfrecord_file) or not tf.io.gfile.exists(eval_tfrecord_file) \
            or not tf.io.gfile.exists(metadata_file):
        dataset.create_features(MAX_SEQ_LENTGH, tokenizer, train_tsv, test_tsv, train_tfrecord_file,
                                eval_tfrecord_file, metadata_file)

    with tf.io.gfile.GFile(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    strategy = None

    if args['tpu']:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args['tpu'])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif args['gpus'].split(',') > 1:
        gpus = [f"/gpu:{gpu}" for gpu in args['gpus'].split(',')]
        strategy = tf.distribute.MirroredStrategy(devices=gpus)
    else:
        gpus = args['gpus'].split(',')
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:" + gpus[0])

    label_list = metadata["labels"]
    train_number_examples = metadata["train_number_examples"]
    batch_tokens = metadata["batch_tokens"]
    batch_size = args['batch_size'] * args['num_tpu_cores'] if args['tpu'] else args['batch_size'] * max(1, len(gpus))
    num_train_steps = int(train_number_examples // batch_size)
    num_train_optimization_steps = num_train_steps * args['epochs']
    eval_drop_remainder = True if args['tpu'] else False
    config = BertConfig.from_pretrained('bert-base-multilingual-cased', num_labels=len(label_list) + 1)
    num_warmup_steps = int(args['warmup_proportion'] * num_train_optimization_steps)

    logging.info('Saving tensorboard summaries at %s', output_dir)
    logging.info('Use TPU at %s', args['tpu'] if args['tpu'] is not None else 'local')

    with strategy.scope():
        logging.info('Building BNER model')

        model = TFBertForTokenClassification.from_pretrained('bert-base-multilingual-cased', config=config)
        model.layers[-1].activation = tf.keras.activations.softmax
        loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        optimizer = optimization.create_optimizer(args['learning_rate'], num_train_optimization_steps, num_warmup_steps)
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
        loss_metric = tf.keras.metrics.Mean()

    logging.info('Compiling BNER model.')
    model.summary()

    train_dataset = file_based_dataset_builder(input_file=train_tfrecord_file,
                                               is_training=True,
                                               drop_remainder=True,
                                               batch_size=batch_size)
    eval_dataset = file_based_dataset_builder(input_file=eval_tfrecord_file,
                                              is_training=False,
                                              drop_remainder=eval_drop_remainder,
                                              batch_size=batch_size)
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    eval_dataset = strategy.experimental_distribute_dataset(eval_dataset)

    @tf.function
    def train_step(features, labels):
        def step_fn(features, labels):
            inputs = {'attention_mask': features['input_mask'], 'token_type_ids': features['segment_ids'], 'training': True}

            with tf.GradientTape() as tape:
                logits = model(features['input_ids'], **inputs)[0]
                logits = tf.reshape(logits,(-1, len(label_list) + 1))
                active_loss = tf.reshape(features['input_mask'], (-1,))
                active_logits = tf.boolean_mask(logits, active_loss)
                labels = tf.reshape(labels,(-1,))
                active_labels = tf.boolean_mask(labels, active_loss)
                cross_entropy = loss_fct(active_labels, active_logits)
                loss = tf.reduce_sum(cross_entropy) * (1.0 / batch_size)

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

            return cross_entropy

        per_example_losses = strategy.experimental_run_v2(step_fn, args=(features, labels))
        mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)

        return mean_loss

    epoch_bar = master_bar(range(args['epochs']))
    current_time = datetime.datetime.now()

    for epoch in epoch_bar:
        with strategy.scope():
            for (features, labels) in progress_bar(train_dataset, total=num_train_steps, parent=epoch_bar):
                loss = train_step(features, labels)
                loss_metric(loss)
                epoch_bar.child.comment = f'loss : {loss_metric.result()}'

            epoch_bar.write(f'loss epoch {epoch}: {loss_metric.result()}')

            loss_metric.reset_states()

    logging.info("  Training took time = {}".format(datetime.datetime.now() - current_time))

    gold_labels, preds = evaluate(model, label_list, eval_dataset)
    output_eval_file = os.path.join(output_dir, "eval_results.txt")

    with tf.io.gfile.GFile(output_eval_file, "w") as writer:
        logging.info("***** Eval results *****")

        report = metrics.classification_report(gold_labels, preds, digits=4)

        logging.info(report)
        writer.write(report)

    output_predict_file = os.path.join(output_dir, "test_results.tsv")

    write_results(output_predict_file, preds, batch_tokens, gold_labels)

    tf.saved_model.save(model, output_dir)
