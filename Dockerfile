FROM ubuntu:18.04

# Environment variables and args

ARG NOTEBOOK_USER=root
ARG NOTEBOOK_UID=1000
ENV USER ${NOTEBOOK_USER}
ENV NOTEBOOK_UID ${NOTEBOOK_UID}
ENV HOME /home/${NOTEBOOK_USER}

WORKDIR ${HOME}

USER root
# Downloads the package lists from the repositories and "updates" them 
# to get information on the newest versions of packages and their dependencies.
RUN apt-get update

# Install 'curl': Command line tool that allows you to transfer data from or to a remote server. 
# With curl, you can download or upload data using HTTP, HTTPS, SCP, SFTP, and FTP.
RUN apt-get install -y curl


### Building image START

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

#Hack for libboost-python binding when both python2 and python3 present.    
RUN cd /usr/lib/x86_64-linux-gnu && \
    unlink libboost_python.so && \
    unlink libboost_python.a && \
    ln -s libboost_python-py36.so libboost_python.so && \
    ln -s libboost_python-py36.a libboost_python.a && \
    cd -

#Start Building
RUN git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe.git . && \
    pip3 install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip3 install $req; done && cd .. && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=1 .. && \
    make -j"$(nproc)" && \
    make runtest

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

### Building image END


# install the notebook package
RUN pip3 install notebook jupyterlab

# Copy notebooks

COPY ./ ${HOME}/Notebooks/

RUN chown -R ${NOTEBOOK_UID} ${HOME}
USER ${USER}


RUN echo "$PATH"

### hack for bug inside Notebooks 
RUN pip3 uninstall -y scipy &&  pip3 install scipy
RUN pip3 uninstall -y pyyaml &&  python3 -m pip install PyYAML

# Set root to Notebooks
WORKDIR ${HOME}/Notebooks/
