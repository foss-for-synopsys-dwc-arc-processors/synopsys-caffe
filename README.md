# Synopsys Caffe

[![Build Status](https://travis-ci.org/foss-for-synopsys-dwc-arc-processors/synopsys-caffe.svg?branch=master)](https://travis-ci.org/foss-for-synopsys-dwc-arc-processors/synopsys-caffe)

Synopsys Caffe is a modified version of the popular [Caffe Deep Learning framework](http://caffe.berkeleyvision.org/) adapted for use with DesignWare EV6x Processors.
It combines multiple customized branches and includes a large range of patches to support diverse models. See [FEATURES.md](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe/blob/master/FEATURES.md) for a short overview.

## Installation
Please check out the prerequisites and read the detailed notes at the [BVLC Caffe Installation](http://caffe.berkeleyvision.org/installation.html) if this is your first time to install Caffe.

### Linux
If you use Ubuntu, you can refer to the [detailed guideline](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe/blob/development/scripts/ubuntu_python3_build_caffe_guide.md) if you want to install the whole EV CNN environment with all the dependencies. 

A simple guide:
1. Ensure that you have all the dependencies mentioned at the [BVLC Caffe Installation](http://caffe.berkeleyvision.org/installation.html) for your OS installed (protobuf, leveldb, snappy, opencv, hdf5-serial, protobuf-compiler, BLAS, Python, CUDA etc.)
2. Also Install [matio](https://github.com/tbeu/matio) in your environment. After that, add [your installed matio path]/lib to the LD_LIBRARY_PATH.
```Shell
export LD_LIBRARY_PATH=[your installed matio path]/lib:${LD_LIBRARY_PATH}
```
3. Checkout the Synopsys Caffe **master** branch. Configure the build by copying and modifying the example Makefile.config for your setup.
```Shell
git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe.git
cd synopsys-caffe
cp Makefile.config.example Makefile.config
# Modify Makefile.config to suit your needs, e.g. enable/disable the CPU-ONLY, CUDNN, NCCL and set the path for CUDA, Python and BLAS.
# If needed, add [your installed matio path]/include to INCLUDE_DIRS and [your installed matio path]/lib to LIBRARY_DIRS.
```
4. Build Caffe and run the tests.
```Shell
make all
make pycaffe
make test
make runtest
# If no error occurs, you can add the caffe path to the environment for easy use.
export SYNOPSYS_CAFFE_HOME=[your synopsys-caffe root folder path]
export PATH=${SYNOPSYS_CAFFE_HOME}/build/tools:${PATH}
export PYTHONPATH=${SYNOPSYS_CAFFE_HOME}/python:${PYTHONPATH}
```


### Windows
A simple guide:
1. Download the **Visual Studio 2015 Update 3** (Do not use the VS 2017, it is not supported!). Choose to install the support for visual C++ instead of applying the default settings.
2. Install the CMake 3.4 or higher. Install Python 2.7 or 3.5/3.6. Add cmake.exe and python.exe to your PATH.
3. After installing the Python, please open a `cmd` prompt and use `pip install numpy` to install the **numpy** package.
4. Checkout the Synopsys Caffe **master** branch for build. The windows branch is deprecated, please do not use it. We use `C:\Projects` as the current folder for the following instructions.
5. Edit any of the options inside **synopsys-caffe\scripts\build_win.cmd** to suit your needs, such as settings for Python version, CUDA/CuDNN enabling etc.   
```cmd
C:\Projects> git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe.git
C:\Projects> cd synopsys-caffe
C:\Projects\synopsys-caffe> scripts\build_win.cmd
:: If no error occurs, the caffe.exe will be created at C:\Projects\synopsys-caffe\build\tools\Release after a successful build.
```
Other detailed installation instructions can be found [here](https://github.com/BVLC/caffe/blob/windows/README.md).

## Reference for different Distributions
Synopsys Caffe support the features introduced in following customized branches. Here are some links to the original demos, tutorials and models usage:
- [SegNet](https://github.com/alexgkendall/caffe-segnet)
- [Faster RCNN](https://github.com/rbgirshick/py-faster-rcnn)
- [SSD](https://github.com/weiliu89/caffe/tree/ssd)
- [FlowNet2](https://github.com/lmb-freiburg/flownet2)
- [SRGAN](https://github.com/ShenghaiRong/caffe_srgan)
- [ICNet (PSPNet)](https://github.com/hszhao/ICNet)

