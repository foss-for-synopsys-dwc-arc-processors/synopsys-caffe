#!/bin/bash
# install dependencies
# (this script must be run as root)
set -x
BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

apt-get -y update
apt-get install -y --no-install-recommends \
  build-essential \
  graphviz \
  libboost-filesystem-dev \
  libboost-python-dev \
  libboost-system-dev \
  libboost-thread-dev \
  libboost-regex-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  libhdf5-serial-dev \
  libopenblas-dev \
  python-virtualenv \
  wget

# Build MatIO
WITH_MATIO=true
if $WITH_MATIO ; then
  MATIO_DIR=~/lib_matio
  pushd .
  if [ -d "$MATIO_DIR" ] && [ -e "$MATIO_DIR/src/libmatio.la" ]; then
    echo "Using cached matio build ..."
    cd $MATIO_DIR
  else
    echo "Building matio from source ..."
    rm -rf $MATIO_DIR
    mkdir $MATIO_DIR

    wget https://github.com/tbeu/matio/archive/v1.5.17.tar.gz -O libmatio.tar.gz
    tar -xzf libmatio.tar.gz -C $MATIO_DIR --strip 1
    rm libmatio.tar.gz
    cd $MATIO_DIR
    ./autogen.sh
    ./configure --prefix=/usr
    make --jobs=$NUM_THREADS
  fi
  make install
  popd
fi
#git clone https://github.com/tbeu/matio.git lib_matio
#cd lib_matio
#git submodule update --init  # for datasets used in unit tests
#./autogen.sh
#./configure
#make --jobs=$NUM_THREADS
##make check  # Check cost a long time
#make install

if $WITH_CMAKE ; then
  apt-get install -y --no-install-recommends cmake
fi

if ! $WITH_PYTHON3 ; then
  # Python2
  apt-get install -y --no-install-recommends \
    libprotobuf-dev \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-protobuf \
    python-pydot \
    python-skimage
else
  # Python3
  apt-get install -y --no-install-recommends \
    python3-dev \
    python3-numpy \
    python3-skimage

  # build Protobuf3 since it's needed for Python3
  PROTOBUF3_DIR=~/protobuf3
  pushd .
  if [ -d "$PROTOBUF3_DIR" ] && [ -e "$PROTOBUF3_DIR/src/protoc" ]; then
    echo "Using cached protobuf3 build ..."
    cd $PROTOBUF3_DIR
  else
    echo "Building protobuf3 from source ..."
    rm -rf $PROTOBUF3_DIR
    mkdir $PROTOBUF3_DIR

    # install some more dependencies required to build protobuf3
    apt-get install -y --no-install-recommends \
      curl \
      dh-autoreconf \
      unzip

    wget https://github.com/protocolbuffers/protobuf/archive/v3.6.1.tar.gz -O protobuf3.tar.gz
    tar -xzf protobuf3.tar.gz -C $PROTOBUF3_DIR --strip 1
    rm protobuf3.tar.gz
    cd $PROTOBUF3_DIR
    ./autogen.sh
    ./configure --prefix=/usr
    make --jobs=$NUM_THREADS
  fi
  make install
  popd
fi

if $WITH_IO ; then
  apt-get install -y --no-install-recommends \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libsnappy-dev
fi

if $WITH_CUDA ; then
  # install repo packages
  CUDA_REPO_PKG=cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/$CUDA_REPO_PKG
  apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

  dpkg -i $CUDA_REPO_PKG
  rm $CUDA_REPO_PKG

  if $WITH_CUDNN ; then
    ML_REPO_PKG=nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/$ML_REPO_PKG
    dpkg -i $ML_REPO_PKG
    rm $ML_REPO_PKG
  fi

  apt-get -y update
  apt-get install cuda-10-0

  if false ; then
    apt-cache search cuda
    apt-cache search libcudnn
    ls -l /usr/local/
    ls -l /usr/lib*
    ls -l /usr/include/
    ls -l /usr/lib/x86_64-linux-gnu
  fi

  if $WITH_CUDNN ; then
    apt-get install -y --no-install-recommends libcudnn7-dev
  fi
fi

