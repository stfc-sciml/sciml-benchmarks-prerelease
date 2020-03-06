FROM nvcr.io/nvidia/tensorflow:20.01-tf2-py3

RUN apt-get update

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ADD . /workspace/
WORKDIR /workspace/

RUN pip install -e . 

CMD sciml-bench
