#!/bin/bash

# Pass extra cmake options to this script, including
# things like -DCMAKE_INSTALL_PREFIX=/path/to/install, etc.

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="clang" \
    -DCMAKE_CXX_COMPILER="clang++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread -std=c++11" \
    -DOPENMP_TARGET_FLAGS="-fopenmp -fopenmp-targets=nvptx64 -fopenmp-target-debug=3 --offload-arch=sm_80 " \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DUSE_OPENMP_TARGET=TRUE \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    ${opts} \
    ..
