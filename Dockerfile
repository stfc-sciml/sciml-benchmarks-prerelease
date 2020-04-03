FROM nvcr.io/nvidia/tensorflow:20.01-tf2-py3

RUN apt-get update

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ADD . /workspace/
WORKDIR /workspace/

RUN pip install -e . 

EXPOSE 5000
ENV SCIML_BENCH_DATA_DIR /data
ENV SCIML_BENCH_MODEL_DIR /out
ENV SCIML_BENCH_TRACKING_URI /out/mlflow

RUN mkdir $SCIML_BENCH_MODEL_DIR
RUN mkdir $SCIML_BENCH_TRACKING_URI

CMD (mlflow server --host 0.0.0.0 --backend-store-uri $SCIML_BENCH_TRACKING_URI --default-artifact-root $SCIML_BENCH_TRACKING_URI &) && sciml-bench all
