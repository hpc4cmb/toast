#!/bin/bash

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="icc" \
    -DCMAKE_CXX_COMPILER="icpc" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -xcore-avx2 -pthread -qno-openmp-offload" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -xcore-avx2 -pthread -qno-openmp-offload" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DUSE_MKL=1 \
    -DDISABLE_OPENMP_TARGET=1 \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${CMBENV_AUX_ROOT}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${CMBENV_AUX_ROOT}/lib" \
    ${opts} \
    ..
