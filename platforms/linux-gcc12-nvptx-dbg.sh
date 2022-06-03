#!/bin/bash

# NOTE:  This assumes gcc >= 12.1.0

# Pass extra cmake options to this script, including
# things like -DCMAKE_INSTALL_PREFIX=/path/to/install, etc.

opts="$@"

cmake \
    -DCMAKE_BUILD_TYPE="Debug" \
    -DCMAKE_C_COMPILER="gcc-12" \
    -DCMAKE_CXX_COMPILER="g++-12" \
    -DCMAKE_C_FLAGS="-O0 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O0 -g -fPIC -pthread -std=c++11 -fcf-protection=none -fno-stack-protector" \
    -DUSE_OPENMP_TARGET=TRUE \
    -DOPENMP_TARGET_FLAGS="\
    -foffload=nvptx-none \
    -foffload-options=-Wa,-m,sm_80 \
    -foffload-options=-misa=sm_80 \
    -foffload-options=-lm \
    -foffload-options=-latomic \
    " \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    ${opts} \
    ..



