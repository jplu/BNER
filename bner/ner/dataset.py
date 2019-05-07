# coding=utf-8

import collections
import pandas as pd
import tensorflow as tf
import bert
from bert import run_classifier
from bert import tokenization
import _pickle as pickle


def _load_dataset(name):
    label_list = ["[PAD]", "X", "[CLS]", "[SEP]"]
    dataset = {"sentence": [], "labels": []}
    tf.logging.info(name + ": " + str(tf.gfile.Exists(name)))
    with tf.gfile.GFile(name) as f:
        words = []
        labels = []
        for line in f:
            contents = line.strip()
            tokens = contents.split(' ')
            if contents.startswith("-DOCSTART-"):
                continue
            if len(tokens) == 2:
                words.append(tokens[0])
                labels.append(tokens[-1])
            else:
                if len(contents) == 0 and len(words) > 0:
                    for l in labels:
                        if l not in label_list:
                            label_list.append(l)
                    dataset["sentence"].append(' '.join(words))
                    dataset["labels"].append(' '.join(labels))
                    words = []
                    labels = []

    return pd.DataFrame.from_dict(dataset), label_list


def load_examples(tsv_file):
    dataset, label_list = _load_dataset(tsv_file)
    dataset_df = pd.concat([dataset]).sample(frac=1).reset_index(drop=True)

    dataset_examples = dataset_df.apply(
        lambda x: bert.run_classifier.InputExample(guid=None, text_a=x["sentence"],
                                                   label=x["labels"]), axis=1)

    return dataset_examples, label_list


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    label_map = {}

    for (i, label) in enumerate(label_list):
        label_map[label] = i

    textlist = example.text_a.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, (word, label) in enumerate(zip(textlist, labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)

        for i, _ in enumerate(token):
            if i == 0:
                labels.append(label)
            else:
                labels.append("X")
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = run_classifier.InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_ids,
    )

    return feature, ntokens, label_ids


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                            output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature, ntokens, label_ids = convert_single_example(ex_index, example, label_list,
                                                             max_seq_length, tokenizer)
        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_id"] = create_int_feature(feature.label_id)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()

    return batch_tokens, batch_labels


def create_features(seq_length, tokenizer, train_tsv, test_tsv, train_tfrecord_file,
                    eval_tfrecord_file, metadata_file):
    if tf.gfile.Exists(train_tfrecord_file):
        tf.gfile.Remove(train_tfrecord_file)

    if tf.gfile.Exists(eval_tfrecord_file):
        tf.gfile.Remove(eval_tfrecord_file)

    if tf.gfile.Exists(metadata_file):
        tf.gfile.Remove(metadata_file)

    train_input_examples, label_list = load_examples(train_tsv)
    eval_input_examples, _ = load_examples(test_tsv)

    _, _ = file_based_convert_examples_to_features(train_input_examples, label_list, seq_length,
                                                   tokenizer, train_tfrecord_file)
    batch_tokens, batch_labels = file_based_convert_examples_to_features(eval_input_examples,
                                                                         label_list, seq_length,
                                                                         tokenizer,
                                                                         eval_tfrecord_file)

    metadata = {"max_seq_len": seq_length, "labels": label_list,
                "train_number_examples": len(train_input_examples),
                "eval_number_examples": len(eval_input_examples), "batch_tokens": batch_tokens,
                "batch_labels": batch_labels}

    with tf.gfile.GFile(metadata_file, "w") as f:
        pickle.dump(metadata, f)
