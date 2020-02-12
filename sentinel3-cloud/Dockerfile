FROM nvcr.io/nvidia/tensorflow:20.01-tf2-py3


RUN apt-get update

COPY requirements.txt /workspace

RUN pip install -r requirements.txt

ADD . /workspace/unet
WORKDIR /workspace/unet

CMD examples/unet_FP32_1GPU.sh . dataset results
