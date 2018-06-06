# Synopsys Caffe    
Synopsys Caffe is a modified version of the popular [Caffe Deep Learning framework](http://caffe.berkeleyvision.org/) adapted for use with DesignWare EV6x Processors.  
It combines multiple customized branches and includes a large range of patches to support diverse models. See [FEATURES.md](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe/blob/master/FEATURES.md) for a short overview.  
  
## Installation  
Please check out the prerequisites and read the detailed notes at the [BVLC Caffe Installation](http://caffe.berkeleyvision.org/installation.html) at first.  
  
### Linux  
Please follow the [instructions](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe/commit/10169e55f4d460c52067792d5f36b9113fa9a705#comments) to install and set the **matio** support before building Caffe.  
### Windows   
A simple guide:
1. Download the Visual Studio 2015 Update 3. Choose to install the support for visual C++ instead of applying the default settings.
2. Install the CMake 3.4 or higher. Install Python 2.7 or 3.5. Add cmake.exe and python.exe to your PATH.
3. After installing the Python, please open a `cmd` prompt and use `pip install numpy` to install the **numpy** package.  
4. Checkout the Synopsys Caffe **master** branch for build. The windows branch is deprecated, please do not use it. We use `C:\Projects` as the current folder for the following instructions.
```cmd
C:\Projects> git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe.git
C:\Projects> cd synopsys-caffe
:: Edit any of the options inside build_win.cmd to suit your needs
C:\Projects\synopsys-caffe> scripts\build_win.cmd
:: If no errors occurs, the caffe.exe will be created at C:\Projects\synopsys-caffe\build\tools\Release after a successful build.
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
