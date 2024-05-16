FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime as base
RUN mkdir -p /opt/project/
WORKDIR /opt/project/
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

from base as app

COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt