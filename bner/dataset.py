# coding=utf-8

import collections
import pandas as pd
import tensorflow as tf
import _pickle as pickle
from absl import logging
from transformers import BertTokenizer


LABELS = []


class InputExample(object):
    def __init__(self, text=None, labels=None):
        # List of tokens
        self.text = text
        # List of labels
        self.labels = labels


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def _load_dataset(name):
    dataset = {"text": [], "labels": []}

    logging.info(name + ": " + str(tf.io.gfile.exists(name)))

    with tf.io.gfile.GFile(name) as f:
        words = []
        labels = []
        for line in f:
            contents = line.strip()
            tokens = contents.split(' ')
            if contents.startswith("-DOCSTART-"):
                continue
            if len(tokens) == 2 or len(tokens) == 4:
                words.append(tokens[0])
                labels.append(tokens[-1])
            else:
                if len(contents) == 0 and len(words) > 0:
                    for l in labels:
                        if l not in LABELS:
                            LABELS.append(l)
                    dataset["text"].append(words)
                    dataset["labels"].append(labels)
                    words = []
                    labels = []
    
    return pd.DataFrame.from_dict(dataset)


def load_examples(tsv_file):
    dataset = _load_dataset(tsv_file)

    dataset_examples = dataset.apply(
        lambda x: InputExample(text=x["text"], labels=x["labels"]), axis=1)

    return dataset_examples


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    label_id_map = {}

    for (i, label) in enumerate(LABELS, 1):
        label_id_map[label] = i

    text = example.text
    labels = example.labels
    tokens = []
    label_ids = []
    
    for word, label in zip(text, labels):
        tokenized_word = tokenizer.tokenize(word)
        
        tokens.extend(tokenized_word)
        label_ids.extend([label_id_map[label]] + [0] * (len(tokenized_word) - 1))

    if len(tokens) >= max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
        label_ids = label_ids[0:(max_seq_length - 2)]
    
    # Add [SEP] token
    tokens.append("[SEP]")
    label_ids.append(0)

    segment_ids = [0] * len(tokens)

    # Add [CLS] token
    tokens = ["[CLS]"] + tokens
    label_ids = [0] + label_ids
    segment_ids = [0] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(tokens)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )

    return feature


def file_based_convert_examples_to_features(examples, max_seq_length, tokenizer,
                                            output_file):
    writer = tf.io.TFRecordWriter(output_file)
    batch_tokens = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer)

        batch_tokens.append(example.text)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        
        writer.write(tf_example.SerializeToString())

    writer.close()

    return batch_tokens


def create_features(max_seq_len, tokenizer, train_tsv, test_tsv, train_tfrecord_file,
                    eval_tfrecord_file, metadata_file):
    global LABELS
    
    if tf.io.gfile.exists(train_tfrecord_file):
        tf.io.gfile.remove(train_tfrecord_file)

    if tf.io.gfile.exists(eval_tfrecord_file):
        tf.io.gfile.remove(eval_tfrecord_file)

    if tf.io.gfile.exists(metadata_file):
        tf.io.gfile.remove(metadata_file)

    train_input_examples = load_examples(train_tsv)
    eval_input_examples = load_examples(test_tsv)

    _ = file_based_convert_examples_to_features(train_input_examples, max_seq_len,
                                                tokenizer, train_tfrecord_file)
    batch_tokens = file_based_convert_examples_to_features(eval_input_examples,
                                                           max_seq_len,
                                                           tokenizer,
                                                           eval_tfrecord_file)

    metadata = {"max_seq_len": max_seq_len, "labels": LABELS,
                "train_number_examples": len(train_input_examples),
                "batch_tokens": batch_tokens}

    with tf.io.gfile.GFile(metadata_file, "w") as f:
        pickle.dump(metadata, f)


def main():
    logging.set_verbosity(logging.INFO)
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
    create_features(128, tokenizer, "../datasets/train.conll", "../datasets/test.conll", "../datasets/train.tf_record",
                    "../datasets/eval.tf_record", "../datasets/metadata.pkl")


if __name__ == "__main__":
    main()
