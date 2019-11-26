FROM jplu/docker-tensorflow-uvicorn-gunicorn-fastapi:python3.7

LABEL maintainer="Julien Plu <julien.plu@eurecom.fr>"

ARG METADATA=./model/metadata.pkl
ARG MODEL_NAME

COPY $METADATA /metadata.pkl

ENV METADATA_FILE /metadata.pkl
ENV MODEL_NAME $MODEL_NAME
ENV APP_MODULE ner:app

RUN pip install --no-cache-dir -U tensorflow-serving-api transformers spacy

COPY ./app /app
