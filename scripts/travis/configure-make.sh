# raw Makefile configuration

LINE () {
  echo "$@" >> Makefile.config
}

cp Makefile.config.example Makefile.config

LINE "BLAS := open"
LINE "WITH_PYTHON_LAYER := 1"

if $WITH_PYTHON3 ; then
  # TODO(lukeyeager) this path is currently disabled because of test errors like:
  #   ImportError: dynamic module does not define init function (PyInit__caffe)
  LINE "PYTHON_LIBRARIES := python3.6m boost_python3-py36"
  LINE "PYTHON_INCLUDE := /usr/include/python3.6 /usr/lib/python3/dist-packages/numpy/core/include"
else
  LINE "PYTHON_LIBRARIES := boost_python python2.7"
fi

LINE "INCLUDE_DIRS := \$(INCLUDE_DIRS) \$(PYTHON_INCLUDE) /usr/include/hdf5/serial/"
LINE "LIBRARY_DIRS := \$(LIBRARY_DIRS) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial"
LINE "PYTHON_LIBRARIES := \$(PYTHON_LIBRARIES) hdf5_serial_hl hdf5_serial"

if ! $WITH_IO ; then
  LINE "USE_OPENCV := 0"
  LINE "USE_LEVELDB := 0"
  LINE "USE_LMDB := 0"
fi
LINE "OPENCV_VERSION := 3"

if $WITH_CUDA ; then
  # Only build SM50
  LINE "CUDA_ARCH := -gencode arch=compute_50,code=sm_50"
else
  LINE "CPU_ONLY := 1"
fi

if $WITH_CUDNN ; then
  LINE "USE_CUDNN := 1"
fi

