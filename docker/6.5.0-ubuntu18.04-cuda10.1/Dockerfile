FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-dev python3-pip \
    && pip3 install setuptools && pip3 install --upgrade pip \
    && apt-get install -y --no-install-recommends git m4 autoconf automake libtool flex \
    && apt-get install -y --no-install-recommends ssh \
    && apt-get install -y --no-install-recommends libopencv-dev \
    && apt-get install -y --no-install-recommends vim \
    && apt-get install -y --no-install-recommends screen \
    && apt-get clean
RUN git clone --depth=1 -b v3.0.0 https://github.com/open-mpi/ompi.git /tmp/ompi \
    && cd /tmp/ompi \
    && ./autogen.pl \
    && ./configure --with-cuda \
    && make -j4 \
    && make install \
    && rm -rf /tmp/ompi

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV CPATH=/usr/local/include:$CPATH
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=$CUDA_PATH/bin:$PATH
ENV CPATH=$CUDA_PATH/include:$CPATH
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:$LD_LIBRARY_PATH

RUN pip3 install cython cupy-cuda101==6.5.0 chainer==6.5.0 chainercv==0.13.1

# install misc packages
RUN pip3 install \
    opencv-python \
    matplotlib \
    yacs \
    tensorflow \
    tensorboardx \
    mpi4py \
    requests \
