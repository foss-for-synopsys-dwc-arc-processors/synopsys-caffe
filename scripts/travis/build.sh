#!/bin/bash
# build the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if ! $WITH_CMAKE ; then
  make --jobs $NUM_THREADS
  make --jobs $NUM_THREADS distribute
  if ! $WITH_CUDA ; then
    make --jobs $NUM_THREADS test pycaffe
  fi
else
  cd build
  make --jobs $NUM_THREADS all test.testbin
fi
# ignore lint
#make lint
