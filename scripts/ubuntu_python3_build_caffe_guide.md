# Special Guide for Installing the whole Environment on Ubuntu (16.04)
Following this guide will build a full set of toolkits with Caffe to provide the necessary SNPS EV cnn_tools dependencies.  
This script will take python3.6 as build environment, and build the cpu-only version.

## Prerequisites
1. gcc version = gfortran version  
If this is not fulfilled, you can use commands like the following to install:
```
sudo apt-get install gcc-7
sudo apt-get install gfortran-7
```
and reset the default path:
```
mkdir -p gcc-symlinks
mkdir -p gfortran-symlinks
ln -s /usr/bin/gcc-7 gcc-symlinks/gcc
ln -s /usr/bin/gfortran-7 gfortran-symlinks/gfortran
export PATH=$PWD/gfortran-symlinks:$PWD/gcc-symlinks:$PATH
```

2. openssl  
Make sure it is installed and its lib path is set correctly:
```
sudo apt-get remove openssl
sudo apt-get install openssl libssl-dev
```

## Build Steps
### Setting Environment

```
export SYNOPSYS_CAFFE_HOME=`pwd`/out

# write the environment setting script
echo "export BINDIR=\"\$SYNOPSYS_CAFFE_HOME/bin\"" > env.sh
echo "export BINLIBDEST=\"\$SYNOPSYS_CAFFE_HOME/lib/python3.6\"" >> env.sh
echo "export BLDSHARED=\"gcc -pthread -shared -L\$SYNOPSYS_CAFFE_HOME/lib -L\$SYNOPSYS_CAFFE_HOME/lib64 -ldl -lutil\"" >> env.sh
echo "export CFLAGS=\"-I\$SYNOPSYS_CAFFE_HOME/include -I\$SYNOPSYS_CAFFE_HOME/include/python3.6m -I\$SYNOPSYS_CAFFE_HOME/lib/python3.6/site-packages/numpy/core/include -fPIC\"" >> env.sh
echo "export CPLUS_INCLUDE_PATH=\"\$SYNOPSYS_CAFFE_HOME/include/python3.6m\"" >> env.sh
echo "export CXXFLAGS=\"\$CFLAGS -std=c++11\"" >> env.sh
echo "export CONFINCLUDEDIR=\"\$SYNOPSYS_CAFFE_HOME/include\"" >> env.sh
echo "export CONFINCLUDEPY=\"\$SYNOPSYS_CAFFE_HOME/include/python3.6m\"" >> env.sh
echo "export DESTDIRS=\"\$SYNOPSYS_CAFFE_HOME \$SYNOPSYS_CAFFE_HOME/lib \$SYNOPSYS_CAFFE_HOME/lib/python3.6 \$SYNOPSYS_CAFFE_HOME/lib/python3.6/lib-dynload\"" >> env.sh
echo "export DESTLIB=\"\$SYNOPSYS_CAFFE_HOME/lib/python3.6\"" >> env.sh
echo "export DESTSHARED=\"\$SYNOPSYS_CAFFE_HOME/lib/python3.6/lib-dynload\"" >> env.sh
echo "export INCLDIRSTOMAKE=\"\$SYNOPSYS_CAFFE_HOME/include \$SYNOPSYS_CAFFE_HOME/include \$SYNOPSYS_CAFFE_HOME/include/python3.6m \$SYNOPSYS_CAFFE_HOME/include/python3.6m\"" >> env.sh
echo "export INCLUDEDIR=\"\$SYNOPSYS_CAFFE_HOME/include\"" >> env.sh
echo "export INCLUDEPY=\"\$SYNOPSYS_CAFFE_HOME/include/python3.6m\"" >> env.sh
echo "export LDFLAGS=\"-L\$SYNOPSYS_CAFFE_HOME/lib -L\$SYNOPSYS_CAFFE_HOME/lib64 -ldl -lutil\"" >> env.sh
echo "export LDSHARED=\"gcc -pthread -shared -L\$SYNOPSYS_CAFFE_HOME/lib -L\$SYNOPSYS_CAFFE_HOME/lib64 -ldl -lutil\"" >> env.sh
echo "export LIBDEST=\"\$SYNOPSYS_CAFFE_HOME/lib/python3.6\"" >> env.sh
echo "export LIBDIR=\"\$SYNOPSYS_CAFFE_HOME/lib\"" >> env.sh
echo "export LIBP=\"\$SYNOPSYS_CAFFE_HOME/lib/python3.6\"" >> env.sh
echo "export LIBPC=\"\$SYNOPSYS_CAFFE_HOME/lib/pkgconfig\"" >> env.sh
echo "export LIBPL=\"\$SYNOPSYS_CAFFE_HOME/lib/python3.6/config\"" >> env.sh
echo "export MACHDESTLIB=\"\$SYNOPSYS_CAFFE_HOME/lib/python3.6\"" >> env.sh
echo "export MANDIR=\"\$SYNOPSYS_CAFFE_HOME/share/man\"" >> env.sh
echo "export PY_CFLAGS=\"-I\$SYNOPSYS_CAFFE_HOME/include -I\$SYNOPSYS_CAFFE_HOME/include/python3.6m -I\$SYNOPSYS_CAFFE_HOME/lib/python3.6/site-packages/numpy/core/include -fPIC\"" >> env.sh
echo "export SCRIPTDIR=\"\$SYNOPSYS_CAFFE_HOME/lib\"" >> env.sh
echo "export datarootdir=\"\$SYNOPSYS_CAFFE_HOME/share\"" >> env.sh
echo "export exec_prefix=\"\$SYNOPSYS_CAFFE_HOME\"" >> env.sh
echo "export prefix=\"\$SYNOPSYS_CAFFE_HOME\"" >> env.sh
echo "export PATH=\"\$SYNOPSYS_CAFFE_HOME/bin:\$PATH\"" >> env.sh
echo "export LD_LIBRARY_PATH=\"\$SYNOPSYS_CAFFE_HOME/lib:\$SYNOPSYS_CAFFE_HOME/lib64:\$LD_LIBRARY_PATH\"" >> env.sh
echo "export PYTHONHOME=\"\$SYNOPSYS_CAFFE_HOME\"" >> env.sh
echo "export PYTHONPATH=\"\$SYNOPSYS_CAFFE_HOME/python:\$SYNOPSYS_CAFFE_HOME/lib:\$PYTHONPATH\"" >> env.sh
chmod +x env.sh

echo "if (! \$?LD_LIBRARY_PATH) then" > env.csh
echo "    setenv LD_LIBRARY_PATH \"\"" >> env.csh
echo "endif" >> env.csh
echo "if (! \$?PYTHONPATH) then" >> env.csh
echo "    setenv PYTHONPATH \"\"" >> env.csh
echo "endif" >> env.csh
echo "setenv BINDIR \"\$SYNOPSYS_CAFFE_HOME/bin\"" >> env.csh
echo "setenv BINLIBDEST \"\$SYNOPSYS_CAFFE_HOME/lib/python3.6\"" >> env.csh
echo "setenv BLDSHARED \"gcc -pthread -shared -L\$SYNOPSYS_CAFFE_HOME/lib -L\$SYNOPSYS_CAFFE_HOME/lib64 -ldl -lutil\"" >> env.csh
echo "setenv CFLAGS \"-I\$SYNOPSYS_CAFFE_HOME/include -I\$SYNOPSYS_CAFFE_HOME/include/python3.6m -I\$SYNOPSYS_CAFFE_HOME/lib/python3.6/site-packages/numpy/core/include -fPIC\"" >> env.csh
echo "setenv CPLUS_INCLUDE_PATH \"\$SYNOPSYS_CAFFE_HOME/include/python3.6m\"" >> env.csh
echo "setenv CXXFLAGS \"\$CFLAGS -std=c++11\"" >> env.csh
echo "setenv CONFINCLUDEDIR \"\$SYNOPSYS_CAFFE_HOME/include\"" >> env.csh
echo "setenv CONFINCLUDEPY \"\$SYNOPSYS_CAFFE_HOME/include/python3.6m\"" >> env.csh
echo "setenv DESTDIRS \"\$SYNOPSYS_CAFFE_HOME \$SYNOPSYS_CAFFE_HOME/lib \$SYNOPSYS_CAFFE_HOME/lib/python3.6 \$SYNOPSYS_CAFFE_HOME/lib/python3.6/lib-dynload\"" >> env.csh
echo "setenv DESTLIB \"\$SYNOPSYS_CAFFE_HOME/lib/python3.6\"" >> env.csh
echo "setenv DESTSHARED \"\$SYNOPSYS_CAFFE_HOME/lib/python3.6/lib-dynload\"" >> env.csh
echo "setenv INCLDIRSTOMAKE \"\$SYNOPSYS_CAFFE_HOME/include \$SYNOPSYS_CAFFE_HOME/include \$SYNOPSYS_CAFFE_HOME/include/python3.6m \$SYNOPSYS_CAFFE_HOME/include/python3.6m\"" >> env.csh
echo "setenv INCLUDEDIR \"\$SYNOPSYS_CAFFE_HOME/include\"" >> env.csh
echo "setenv INCLUDEPY \"\$SYNOPSYS_CAFFE_HOME/include/python3.6m\"" >> env.csh
echo "setenv LDFLAGS \"-L\$SYNOPSYS_CAFFE_HOME/lib -L\$SYNOPSYS_CAFFE_HOME/lib64 -ldl -lutil\"" >> env.csh
echo "setenv LDSHARED \"gcc -pthread -shared -L\$SYNOPSYS_CAFFE_HOME/lib -L\$SYNOPSYS_CAFFE_HOME/lib64 -ldl -lutil\"" >> env.csh
echo "setenv LIBDEST \"\$SYNOPSYS_CAFFE_HOME/lib/python3.6\"" >> env.csh
echo "setenv LIBDIR \"\$SYNOPSYS_CAFFE_HOME/lib\"" >> env.csh
echo "setenv LIBP \"\$SYNOPSYS_CAFFE_HOME/lib/python3.6\"" >> env.csh
echo "setenv LIBPC \"\$SYNOPSYS_CAFFE_HOME/lib/pkgconfig\"" >> env.csh
echo "setenv LIBPL \"\$SYNOPSYS_CAFFE_HOME/lib/python3.6/config\"" >> env.csh
echo "setenv MACHDESTLIB \"\$SYNOPSYS_CAFFE_HOME/lib/python3.6\"" >> env.csh
echo "setenv MANDIR \"\$SYNOPSYS_CAFFE_HOME/share/man\"" >> env.csh
echo "setenv PY_CFLAGS \"-I\$SYNOPSYS_CAFFE_HOME/include -I\$SYNOPSYS_CAFFE_HOME/include/python3.6m -I\$SYNOPSYS_CAFFE_HOME/lib/python3.6/site-packages/numpy/core/include -fPIC\"" >> env.csh
echo "setenv SCRIPTDIR \"\$SYNOPSYS_CAFFE_HOME/lib\"" >> env.csh
echo "setenv datarootdir \"\$SYNOPSYS_CAFFE_HOME/share\"" >> env.csh
echo "setenv exec_prefix \"\$SYNOPSYS_CAFFE_HOME\"" >> env.csh
echo "setenv prefix \"\$SYNOPSYS_CAFFE_HOME\"" >> env.csh
echo "setenv PATH \"\$SYNOPSYS_CAFFE_HOME/bin:\$PATH\"" >> env.csh
echo "setenv LD_LIBRARY_PATH \"\$SYNOPSYS_CAFFE_HOME/lib:\$SYNOPSYS_CAFFE_HOME/lib64:\$LD_LIBRARY_PATH\"" >> env.csh
echo "setenv PYTHONHOME \"\$SYNOPSYS_CAFFE_HOME\"" >> env.csh
echo "setenv PYTHONPATH \"\$SYNOPSYS_CAFFE_HOME/python:\$SYNOPSYS_CAFFE_HOME/lib:\$PYTHONPATH\"" >> env.csh
chmod +x env.csh

# Wipe and re-create directories
rm -rf build
rm -rf out
rm -rf distro
mkdir -p build
mkdir -p out
mkdir -p distro

# use bash environment for build (you can also use csh, then need to source env.csh isntead)
source env.sh
```

### Compiling Dependencies
```
# CMake: https://cmake.org/
if [ ! -f distro/cmake.tar.gz ]; then
    wget -O distro/cmake.tar.gz https://cmake.org/files/v3.9/cmake-3.9.3.tar.gz
fi
tar zxf distro/cmake.tar.gz -C build
cd build/cmake-3.9.3
./bootstrap --prefix=$SYNOPSYS_CAFFE_HOME
make
make install
cd ../..

# Tcl
if [ ! -f distro/tcl.tar.gz ]; then
    wget -O distro/tcl.tar.gz https://prdownloads.sourceforge.net/tcl/tcl8.6.8-src.tar.gz
fi
tar zxf distro/tcl.tar.gz -C build
cd build/tcl8.6.8/unix
./configure --prefix=$SYNOPSYS_CAFFE_HOME
make
make install
cd ../../..

# Tk
if [ ! -f distro/tk.tar.gz ]; then
    wget -O distro/tk.tar.gz https://prdownloads.sourceforge.net/tcl/tk8.6.8-src.tar.gz
fi
tar zxf distro/tk.tar.gz -C build
cd build/tk8.6.8/unix
./configure --prefix=$SYNOPSYS_CAFFE_HOME
make
make install
cd ../../..

# Python: https://www.python.org/
# please make sure you have installed the openssl packages correctly,
# otherwise you may have certificate issues in pip install in the future
sudo apt-get install libssl-dev
sudo apt-get remove openssl
sudo apt-get install openssl

if [ ! -f distro/python.tar.gz ]; then
    wget -O distro/python.tar.gz https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tgz
fi
tar zxf distro/python.tar.gz -C build
cd build/Python-3.6.5
./configure --prefix=$SYNOPSYS_CAFFE_HOME --enable-shared --with-ensurepip=install --enable-unicode=ucs4 --with-tcltk-includes="-I${SYNOPSYS_CAFFE_HOME}/include" --with-tcltk-libs="-L${SYNOPSYS_CAFFE_HOME}/lib"
make
make install
cd ../..
cp -f env.sh $SYNOPSYS_CAFFE_HOME
cp -f env.csh $SYNOPSYS_CAFFE_HOME

# ProtoBuf: https://github.com/google/protobuf https://developers.google.com/protocol-buffers/
if [ ! -f distro/protobuf.tar.gz ]; then
    wget -O distro/protobuf.tar.gz https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/protobuf-cpp-3.7.1.tar.gz
fi
tar -vzxf distro/protobuf.tar.gz -C build
cd build/protobuf-3.7.1
./configure --prefix=$SYNOPSYS_CAFFE_HOME
make
make install
cd ../..

# Snappy: https://github.com/google/snappy and https://google.github.io/snappy/
if [ ! -f distro/snappy.tar.gz ]; then
    wget -O distro/snappy.tar.gz https://github.com/google/snappy/archive/1.1.7.tar.gz
fi
tar zxf distro/snappy.tar.gz -C build
cd build/snappy-1.1.7
mkdir result
cd result
# add -DSNAPPY_BUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Release specially for Ubuntu,
# otherwise may cause errors
cmake -DSNAPPY_BUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$SYNOPSYS_CAFFE_HOME ..
make
make install
cd ../../..

# LevelDB: http://leveldb.org/ and https://github.com/google/leveldb
if [ ! -f distro/leveldb.tar.gz ]; then
    wget -O distro/leveldb.tar.gz https://github.com/google/leveldb/archive/v1.20.tar.gz
fi
tar zxf distro/leveldb.tar.gz -C build
cd build/leveldb-1.20
make
cp -f out-shared/libleveldb.so* $SYNOPSYS_CAFFE_HOME/lib
cp -f out-static/lib*.a $SYNOPSYS_CAFFE_HOME/lib
cp -rf include $SYNOPSYS_CAFFE_HOME
cd ../..

# Boost: http://www.boost.org/
if [ ! -f distro/boost.tar.gz ]; then
    wget -O distro/boost.tar.gz https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz
fi
tar zxf distro/boost.tar.gz -C build
cd build/boost_1_65_1
./bootstrap.sh --prefix=$SYNOPSYS_CAFFE_HOME --with-python=python3
./b2 install --prefix=$SYNOPSYS_CAFFE_HOME
cd ../..

# HDF5: https://support.hdfgroup.org/HDF5/
if [ ! -f distro/szip.tar.gz ]; then
    wget -O distro/szip.tar.gz https://support.hdfgroup.org/ftp/lib-external/szip/2.1.1/src/szip-2.1.1.tar.gz
fi
tar zxf distro/szip.tar.gz -C build
cd build/szip-2.1.1
mkdir result
cd result
cmake -DCMAKE_INSTALL_PREFIX=$SYNOPSYS_CAFFE_HOME ..
make
make install
cd ../../..

# MatIO: https://sourceforge.net/projects/matio/
if [ ! -f distro/matio.tar.gz ]; then
    wget -O distro/matio.tar.gz https://sourceforge.net/projects/matio/files/matio/1.5.12/matio-1.5.12.tar.gz
fi
tar zxf distro/matio.tar.gz -C build
cd build/matio-1.5.12
./configure --prefix=$SYNOPSYS_CAFFE_HOME
make
make install
cd ../..

# HDF5: https://support.hdfgroup.org/HDF5/
if [ ! -f distro/hdf5.tar.gz ]; then
    wget -O distro/hdf5.tar.gz https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.5.tar.gz
fi
tar zxf distro/hdf5.tar.gz -C build
cd build/hdf5-1.10.5
./configure --prefix=$SYNOPSYS_CAFFE_HOME
make
make install
cd ../..

# GFlags: https://github.com/gflags
if [ ! -f distro/gflags.tar.gz ]; then
    wget -O distro/gflags.tar.gz https://github.com/gflags/gflags/archive/v2.2.1.tar.gz
fi
tar zxf distro/gflags.tar.gz -C build
cd build/gflags-2.2.1
mkdir result
cd result
cmake -DCMAKE_INSTALL_PREFIX=$SYNOPSYS_CAFFE_HOME -DCMAKE_CXX_FLAGS:STRING=-fPIC ..
make
make install
cd ../../..

# LMDB: https://github.com/LMDB/lmdb and https://symas.com/lmdb/technical/
if [ ! -f distro/lmdb.tar.gz ]; then
    wget -O distro/lmdb.tar.gz https://github.com/LMDB/lmdb/archive/LMDB_0.9.21.tar.gz
fi
tar zxf distro/lmdb.tar.gz -C build
cd build/lmdb-LMDB_0.9.21/libraries/liblmdb
make
make prefix=$SYNOPSYS_CAFFE_HOME install
cd ../../../..

# GLog: https://github.com/google/glog
if [ ! -f distro/glog.tar.gz ]; then
    wget -O distro/glog.tar.gz https://github.com/google/glog/archive/v0.3.5.tar.gz
fi
tar zxf distro/glog.tar.gz -C build
cd build/glog-0.3.5
./configure --prefix=$SYNOPSYS_CAFFE_HOME
make
make install
cd ../..

# OpenBLAS: http://www.openblas.net/
# please make sure your gcc and gfortran version fulfill the requests
if [ ! -f distro/openblas.tar.gz ]; then
    wget -O distro/openblas.tar.gz http://github.com/xianyi/OpenBLAS/archive/v0.2.20.tar.gz
fi
tar zxf distro/openblas.tar.gz -C build
cd build/OpenBLAS-0.2.20
make DYNAMIC_ARCH=1
make PREFIX=$SYNOPSYS_CAFFE_HOME install
cd ../..

# OpenCV: http://opencv.org/
if [ ! -f distro/opencv-2.zip ]; then
    wget -O distro/opencv-2.zip https://github.com/opencv/opencv/archive/2.4.13.6.zip
fi
unzip -q -o distro/opencv-2.zip -d build
cd build/opencv-2.4.13.6
mkdir result
cd result
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$SYNOPSYS_CAFFE_HOME -D WITH_CUDA=OFF -D WITH_CUFFT=OFF -D WITH_QT=OFF -D WITH_GTK=OFF -DBUILD_opencv_java=OFF ..
make
make install
cd ../../..
```

### Build Synopsys Caffe
```
# Caffe: http://caffe.berkeleyvision.org/ and https://github.com/BVLC/caffe and http://caffe.berkeleyvision.org/installation.html
cd build
git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe.git -b master
cd synopsys-caffe
sed -i -r "s/^(COMMON_FLAGS \+= -DCAFFE_VERSION=).*$/\1${TARGET_VERSION}-${BUILD_TYPE}/" Makefile
cp Makefile.config.example Makefile.config

# modify the Makefile.config settings
echo "--- Makefile.config 2018-02-13 14:35:28.000000000 +0300" > Makefile.config.patch
echo "+++ Makefile.config 2018-02-13 14:35:29.000000000 +0300" >> Makefile.config.patch
echo "@@ -5,7 +5,7 @@" >> Makefile.config.patch
echo " # USE_CUDNN := 1" >> Makefile.config.patch
echo " " >> Makefile.config.patch
echo " # CPU-only switch (uncomment to build without GPU support)." >> Makefile.config.patch
echo "-# CPU_ONLY := 1" >> Makefile.config.patch
echo "+CPU_ONLY := 1" >> Makefile.config.patch
echo " " >> Makefile.config.patch
echo " # uncomment to disable IO dependencies and corresponding data layers" >> Makefile.config.patch
echo " # USE_OPENCV := 0" >> Makefile.config.patch    
echo "@@ -67,8 +67,8 @@ BLAS := open" >> Makefile.config.patch
echo " " >> Makefile.config.patch
echo " # NOTE: this is required only if you will compile the python interface." >> Makefile.config.patch    
echo " # We need to be able to find Python.h and numpy/arrayobject.h." >> Makefile.config.patch
echo "-PYTHON_INCLUDE := /usr/include/python2.7 \\" >> Makefile.config.patch
echo "-               /usr/lib/python2.7/dist-packages/numpy/core/include" >> Makefile.config.patch
echo "+#PYTHON_INCLUDE := /usr/include/python2.7 \\" >> Makefile.config.patch
echo "+#              /usr/lib/python2.7/dist-packages/numpy/core/include" >> Makefile.config.patch
echo " # Anaconda Python distribution is quite popular. Include path:" >> Makefile.config.patch
echo " # Verify anaconda location, sometimes it's in root." >> Makefile.config.patch
echo " # ANACONDA_HOME := \$(HOME)/anaconda2" >> Makefile.config.patch
echo "@@ -77,12 +77,12 @@ PYTHON_INCLUDE := /usr/include/python2.7 \\" >> Makefile.config.patch
echo "                # \$(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include" >> Makefile.config.patch
echo " " >> Makefile.config.patch
echo " # Uncomment to use Python 3 (default is Python 2)" >> Makefile.config.patch
echo "-# PYTHON_LIBRARIES := boost_python3 python3.5m" >> Makefile.config.patch
echo "-# PYTHON_INCLUDE := /usr/include/python3.5m \\" >> Makefile.config.patch
echo "-#                 /usr/lib/python3.5/dist-packages/numpy/core/include" >> Makefile.config.patch
echo "+PYTHON_LIBRARIES := boost_python3 python3.6m" >> Makefile.config.patch
echo "+PYTHON_INCLUDE := \$(SYNOPSYS_CAFFE_HOME)/include/python3.6m \\" >> Makefile.config.patch
echo "+                 \$(SYNOPSYS_CAFFE_HOME)/lib/python3.6/site-packages/numpy/core/include" >> Makefile.config.patch
echo " " >> Makefile.config.patch
echo " # We need to be able to find libpythonX.X.so or .dylib." >> Makefile.config.patch
echo "-PYTHON_LIB := /usr/lib" >> Makefile.config.patch
echo "+PYTHON_LIB := \$(SYNOPSYS_CAFFE_HOME)/lib" >> Makefile.config.patch
echo " # PYTHON_LIB := \$(ANACONDA_HOME)/lib" >> Makefile.config.patch
echo " " >> Makefile.config.patch
echo " # Homebrew installs numpy in a non standard path (keg only)" >> Makefile.config.patch
echo "@@ -93,8 +93,8 @@ PYTHON_LIB := /usr/lib" >> Makefile.config.patch
echo " WITH_PYTHON_LAYER := 1" >> Makefile.config.patch
echo " " >> Makefile.config.patch
echo " # Whatever else you find you need goes here." >> Makefile.config.patch
echo "-INCLUDE_DIRS := \$(PYTHON_INCLUDE) /usr/local/include" >> Makefile.config.patch
echo "-LIBRARY_DIRS := \$(PYTHON_LIB) /usr/local/lib /usr/lib" >> Makefile.config.patch
echo "+INCLUDE_DIRS := \$(PYTHON_INCLUDE) \$(SYNOPSYS_CAFFE_HOME)/include /usr/local/include" >> Makefile.config.patch
echo "+LIBRARY_DIRS := \$(PYTHON_LIB) \$(SYNOPSYS_CAFFE_HOME)/lib /usr/local/lib /usr/lib" >> Makefile.config.patch
echo " " >> Makefile.config.patch
echo " # If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies" >> Makefile.config.patch
echo " # INCLUDE_DIRS += \$(shell brew --prefix)/include" >> Makefile.config.patch
patch -p0 -l < Makefile.config.patch

echo "--- CMakeLists.txt 2019-07-21 14:35:28.000000000 +0300" > CMakeLists.txt.patch
echo "+++ CMakeLists.txt 2019-07-21 14:35:29.000000000 +0300" >> CMakeLists.txt.patch
echo "@@ -47,7 +47,7 @@ else()" >> CMakeLists.txt.patch
echo "   caffe_option(BUILD_SHARED_LIBS \"Build shared libraries\" ON)" >> CMakeLists.txt.patch
echo " endif()" >> CMakeLists.txt.patch
echo " caffe_option(BUILD_python \"Build Python wrapper\" ON)" >> CMakeLists.txt.patch
echo "-set(python_version \"2\" CACHE STRING \"Specify which Python version to use\")" >> CMakeLists.txt.patch
echo "+set(python_version \"3\" CACHE STRING \"Specify which Python version to use\")" >> CMakeLists.txt.patch
echo " caffe_option(BUILD_matlab \"Build Matlab wrapper\" OFF)" >> CMakeLists.txt.patch
echo " caffe_option(BUILD_docs   \"Build documentation\" ON IF UNIX OR APPLE)" >> CMakeLists.txt.patch
echo " caffe_option(BUILD_python_layer \"Build the Caffe Python layer\" ON)" >> CMakeLists.txt.patch
patch -p0 < CMakeLists.txt.patch

# install the necessary python packages
# if you encounter SSL certificate errors, you may need to reinstall the OpenSSL and Python
pip3 install --upgrade pip
for req in $(cat python/requirements.txt); do pip3 install $req --no-cache-dir; done
pip3 install Jinja2>=2.8
pip3 install MarkupSafe>=0.23
pip3 install Pillow>=3.2.0
pip3 install numpy==1.15.4
pip3 install EasyDict
pip3 install sqlalchemy
pip3 install sklearn
pip3 install future
pip3 install sympy
pip3 install opencv-python
pip3 install pycocotools
pip3 install tqdm
pip3 install configparser
pip3 install prettytable
pip3 install pyexcel-xls pyexcel
pip3 install tensorflow==1.12.0
pip3 install onnx
pip3 install protobuf==3.7.1
pip3 install pandas==0.24.0

cd src
protoc caffe/proto/caffe.proto --cpp_out=$SYNOPSYS_CAFFE_HOME/include/
cd ..
export LDFLAGS="$CFLAGS $LDFLAGS"
export NVCCFLAGS="-I$SYNOPSYS_CAFFE_HOME/include -I$SYNOPSYS_CAFFE_HOME/include/python3.6m -I$SYNOPSYS_CAFFE_HOME/lib/python3.6/site-packages/numpy/core/include"
export CFLAGS="$CFLAGS -I/usr/local/cuda/include"
export CXXFLAGS="$CFLAGS -I/usr/local/cuda/include"
export PATH="$PATH:/usr/local/cuda/bin"
make -j4 all
make -j4 test
make distribute
cp -rf distribute/* $SYNOPSYS_CAFFE_HOME
cp -rf models scripts examples $SYNOPSYS_CAFFE_HOME
mkdir $SYNOPSYS_CAFFE_HOME/include/gtest
cp src/gtest/gtest.h $SYNOPSYS_CAFFE_HOME/include/gtest
make runtest
make pycaffe
cd ../..

# Output tool and library versions
gcc --version > versions.txt
ls -1 build/ >> versions.txt
pip3 freeze >> versions.txt

# Update shebang
find $SYNOPSYS_CAFFE_HOME -type f -exec sed -i "1 s/^#!${SYNOPSYS_CAFFE_HOME//\//\\/}\/bin\//#!\/usr\/bin\/env /" '{}' \;

# vefify whether the built Caffe work
python3 -c 'import caffe'
caffe.bin -help || true
caffe.bin -version || true

echo "Success!"
```
