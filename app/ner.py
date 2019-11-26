# coding=utf-8
import grpc
import os
from fastapi import FastAPI
from pydantic import BaseModel
import _pickle as pickle
import tensorflow as tf
import tensorflow_text as text
from spacy.tokens import Span
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.pipeline import Sentencizer
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from starlette.middleware.cors import CORSMiddleware
from absl import logging
import numpy as np
from transformers import BertTokenizer


class Ad(BaseModel):
    text: str


logging.set_verbosity(logging.ERROR)

app = FastAPI()

# CORS
origins = ["*"]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

METADATA_FILE = os.environ.get("METADATA_FILE")
MODEL_NAME = os.environ.get("MODEL_NAME")
HOST_NAME = os.environ.get("HOST_NAME") if os.environ.get("HOST_NAME") else "localhost"
GRPC_SERVER_PORT = os.environ.get("GRPC_SERVER_PORT") if os.environ.get("GRPC_SERVER_PORT") else 8500

with tf.io.gfile.GFile(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

LABELS = metadata["labels"]
MAX_SEQ_LENGTH = metadata['max_seq_len']
tokenizer = text.UnicodeScriptTokenizer()
tokenizer_transformer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
sentencizer = Sentencizer(punct_chars=['𑙂', '᱾', '᭛', '𑁇', '꩝', '𑱁', '𑗔', '𑪛', '𑂾', '᱿', '⸼', '፨', '։', '𖩯', '𑈸', '𑁈', '۔', '𑗊', '꫱', '𑇅', '?', '𑇍', '𑊩', '᪫', '𐩖', '𖫵', '‼', '﹒', '𑅁', '᭚', '𑩂', '𑂿', '﹖', '꧉', '၊', '𑗕', '᙮', '𑗓', '᥄', '𑅂', '．', '𖬷', '𑗂', '᪪', '᭟', '॥', '𑑌', '।', '᪩', '！', '𑗏', '𑗍', '꘏', '𑗑', '᥅', '܀', '𑗖', '𑗗', '𑜽', '꧈', '𐩗', '𑇟', '꛳', '᪨', '᰻', '᜶', '܁', '꩞', '𑑋', '꫰', '⸮', '꯫', '꣎', '𑃁', '𑗉', '᠃', '.', '𝪈', '؟', '᠉', '꛷', '𑗋', '𑇞', '꣏', '𑃀', '𑗐', '။', '᭞', '።', '꡷', '꡶', '𖩮', '᜵', '？', '𑇆', '᰼', '꘎', '﹗', '܂', '፧', '߹', '꓿', '꩟', '⁇', '‽', '𑩃', '𑙁', '!', '𑅃', '𑜼', '𑪜', '⁉', '꤯', '𑱂', '𑈹', '𖬸', '𛲟', '𑜾', '𑈼', '𑗃', '𑗎', '⁈', '𑗌', '𑗒', '𑈻', '𖭄'])
channel = grpc.insecure_channel(HOST_NAME + ":" + str(GRPC_SERVER_PORT))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


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


def convert_single_example(example):
    label_id_map = {}

    for (i, label) in enumerate(LABELS, 1):
        label_id_map[label] = i

    text = example.text
    labels = example.labels
    tokens = []
    label_ids = []
    
    for word, label in zip(text, labels):
        tokenized_word = tokenizer_transformer.tokenize(word)
        
        tokens.extend(tokenized_word)
        label_ids.extend([label_id_map[label]] + [0] * (len(tokenized_word) - 1))

    if len(tokens) >= MAX_SEQ_LENGTH - 2:
        tokens = tokens[0:(MAX_SEQ_LENGTH - 2)]
        label_ids = label_ids[0:(MAX_SEQ_LENGTH - 2)]
    
    # Add [SEP] token
    tokens.append("[SEP]")
    label_ids.append(0)

    segment_ids = [0] * len(tokens)

    # Add [CLS] token
    tokens = ["[CLS]"] + tokens
    label_ids = [0] + label_ids
    segment_ids = [0] + segment_ids

    input_ids = tokenizer_transformer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(tokens)

    while len(input_ids) < MAX_SEQ_LENGTH:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == MAX_SEQ_LENGTH
    assert len(input_mask) == MAX_SEQ_LENGTH
    assert len(segment_ids) == MAX_SEQ_LENGTH
    assert len(label_ids) == MAX_SEQ_LENGTH

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


@app.get("/api/health")
async def health():
    return {"message": "I'm alive"}


@app.post("/api/recognize")
async def get_prediction(ad: Ad):
    (tokens, offset_starts, offset_limits) = tokenizer.tokenize_with_offsets([ad.text])
    spaces = [not offset_limits.to_list()[0][i-1] == offset_starts.to_list()[0][i] for i in range(1,len(tokens[0]))] + [False]
    tokens = [tok.decode() for tok in tokens.to_list()[0]]
    vocab = Vocab(strings=tokens)
    doc = Doc(vocab, words=tokens, spaces=spaces)
    input_example = InputExample(text=tokens, labels=['O'] * len(tokens))
    feature = convert_single_example(input_example)
    model_request = predict_pb2.PredictRequest()
    model_request.model_spec.name = MODEL_NAME
    model_request.model_spec.signature_name = 'serving_default'
    input_ids = tf.make_tensor_proto(feature.input_ids, shape=[1, MAX_SEQ_LENGTH], dtype=tf.int64)
    model_request.inputs['input_1'].CopyFrom(input_ids)
    result = stub.Predict(model_request, 5.0)
    result = tf.make_ndarray(result.outputs["output_1"])

    return output(doc, np.argmax(result[0], -1))


@app.post("/api/tokenize")
async def get_tokens(ad: Ad):
    (tokens, offset_starts, offset_limits) = tokenizer.tokenize_with_offsets([ad.text])
    spaces = [not offset_limits.to_list()[0][i-1] == offset_starts.to_list()[0][i] for i in range(1,len(tokens[0]))] + [False]
    tokens = [tok.decode() for tok in tokens.to_list()[0]]
    vocab = Vocab(strings=tokens)
    doc = Doc(vocab, words=tokens, spaces=spaces)
    sentencizer(doc)
    doc = [[token for token in sent] for sent in doc.sents]
    sentences = []

    for sentence in doc:
        tokens = []

        for token in sentence:
            tokens.append({"value": token.text, "begin": token.idx, "end": token.idx + len(token.text)})

        sentences.append({"tokens": tokens})
    
    return {"sentences": sentences}


def output(doc, ids):
    res = {"entities": []}
    entities = []
    logging.info(ids)
    logging.info(LABELS)
    annotations = list(filter(lambda a: a != 0, ids))
    annotations = annotations[0:len(doc)]
    logging.info(list(doc))
    logging.info([LABELS[label_id - 1] for label_id in annotations])
    prev_type = LABELS[annotations[0] - 1]
    start_span = 0
    end_span = 1

    for idx in range(1, len(annotations)):
        if prev_type != LABELS[annotations[idx] - 1] and prev_type != 'O':
            entities.append({"type": prev_type, "start": start_span, "end": end_span})
            prev_type = LABELS[annotations[idx] - 1]
            start_span = idx
            end_span = idx + 1
        elif annotations[idx] != 'O' and prev_type != 'O':
            end_span += 1
        else:
            prev_type = LABELS[annotations[idx] - 1]
            start_span = idx
            end_span = idx + 1

    if prev_type != 'O':
        entities.append({"type": prev_type, "start": start_span, "end": end_span})

    for ent in entities:
        span = Span(doc, ent["start"], ent["end"], label=ent["type"])
        doc.ents = list(doc.ents) + [span]

    for ent in doc.ents:
        res["entities"].append({"phrase": ent.text, "type": ent.label_,
                                "startOffset": ent.start_char, "endOffset": ent.end_char})

    return res
