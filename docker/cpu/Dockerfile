FROM ubuntu:18.04
LABEL maintainer justin.you@synopsys.com

# workaround bug https://grigorkh.medium.com/fix-tzdata-hangs-docker-image-build-cdb52cc3360d
ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt install -y tzdata


RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        python3.8-dev \
        python3-skimage \
        python3-opencv \
        python3-pip \
        #required by pandas
        libgfortran5 \
        libopenblas-dev \
        libatlas-base-dev \
        libboost-python1.65-dev \
        libboost-all-dev \
        libgflags-dev \
        #Directly incorporate Google glog projects from Github instead of consume it.
        #See https://github.com/google/glog#incorporating-glog-into-a-cmake-project
        #Not install google glog into env and incorporate into cmake build directly.
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        libmatio-dev \
        protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

#update cmake version from default 3.10 to latest
RUN pip3 install --upgrade pip && \
    pip3 install --upgrade cmake && \
    cmake --version

#remove python2 and only keep python3
RUN apt remove python-dev && \
    apt autoremove --purge && \
    which python

RUN git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe.git . && \
    pip3 install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip3 install $req; done && cd .. && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=1 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
