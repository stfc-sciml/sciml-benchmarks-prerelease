FROM nvcr.io/nvidia/tensorflow:19.06-py3


RUN apt-get update
# RUN apt-get install -y libhdf5-serial-dev
# ENV USE_SETUPCFG=0
# ENV HDF5_INCDIR=/usr/include/hdf5/serial
# ENV HDF5_LIBDIR=/usr/lib/x86_64-linux-gnu/hdf5/serial
# ENV HDF5_USE_FILE_LOCKING=FALSE 

COPY requirements.txt /workspace

RUN pip install -r requirements.txt
# RUN pip install netcdf4
# RUN pip uninstall -y h5py

ADD . /workspace/unet
WORKDIR /workspace/unet

CMD examples/unet_FP32_1GPU.sh . dataset results
