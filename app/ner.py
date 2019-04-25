# coding=utf-8
import grpc
import os
import bert
from bert import run_classifier
from bert import tokenization
from fastapi import FastAPI
from pydantic import BaseModel
import _pickle as pickle
import tensorflow as tf
import spacy
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow_hub as hub
from starlette.middleware.cors import CORSMiddleware


BERT_MODEL_HUB = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"

spacy.prefer_gpu()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    label_map = {}

    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i

    textlist = example.text_a.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
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
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

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

    return feature


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


class Ad(BaseModel):
    text: str


# tf.logging.set_verbosity(tf.logging.INFO)

app = FastAPI()

# CORS
origins = ["*"]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

METADATA_FILE = os.environ.get("METADATA_FILE")
MODEL_NAME = os.environ.get("MODEL_NAME")

with tf.io.gfile.GFile(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

label_list = metadata["labels"]
MAX_SEQ_LENGTH = metadata["max_seq_len"]
tokenizer = create_tokenizer_from_hub_module()
nlp = spacy.load('fr_core_news_md')


@app.get("/api/ner/health")
async def health():
    return {"message": "I'm alive"}


@app.post("/api/ner/recognize")
async def get_prediction(ad: Ad):
    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    txt = " ".join([token.text for token in nlp.tokenizer(ad.text)])
    input_example = bert.run_classifier.InputExample(guid="", text_a=txt, label=" ".join(['X'] * len(txt)))
    feature = convert_single_example(0, input_example, label_list, MAX_SEQ_LENGTH, tokenizer)
    model_request = predict_pb2.PredictRequest()
    model_request.model_spec.name = MODEL_NAME
    model_request.model_spec.signature_name = 'serving_default'
    input_ids = tf.contrib.util.make_tensor_proto(feature.input_ids, shape=[1, MAX_SEQ_LENGTH])
    input_mask = tf.contrib.util.make_tensor_proto(feature.input_mask, shape=[1, MAX_SEQ_LENGTH])
    label_id = tf.contrib.util.make_tensor_proto(feature.label_id, shape=[1, MAX_SEQ_LENGTH])
    segment_ids = tf.contrib.util.make_tensor_proto(feature.segment_ids, shape=[1, MAX_SEQ_LENGTH])
    model_request.inputs['input_ids'].CopyFrom(input_ids)
    model_request.inputs['input_mask'].CopyFrom(input_mask)
    model_request.inputs['label_id'].CopyFrom(label_id)
    model_request.inputs['segment_ids'].CopyFrom(segment_ids)
    result = stub.Predict(model_request, 5.0)
    result = tf.make_ndarray(result.outputs["output"])

    return output(txt.split(), result[0])


def output(tokens, ids):
    res = []

    annotations = list(filter(lambda a: a != 0 and a != label_list.index('X'), ids))[1:]

    for token, annotation in zip(tokens, annotations):
        res.append({"word": token, "label": label_list[annotation]})

    return res
