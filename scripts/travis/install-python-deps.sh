#!/bin/bash
# install extra Python dependencies
# (must come after setup-venv)

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if ! $WITH_PYTHON3 ; then
  # Python2
  :
else
  # Python3
  pip install protobuf==3.6.1
  pip install pydot
fi
python --version
pip --version
