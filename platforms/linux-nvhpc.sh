#!/bin/bash

# Pass extra cmake options to this script, including
# things like -DCMAKE_INSTALL_PREFIX=/path/to/install, etc.

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="nvc" \
    -DCMAKE_CXX_COMPILER="nvc++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread -mp=gpu -Minfo=mp -gpu=cc86" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread -std=c++11 -mp=gpu -Minfo=mp -gpu=cc86 -cuda" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    ${opts} \
    ..
