#!/bin/bash

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="icc" \
    -DCMAKE_CXX_COMPILER="icpc" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -xcore-avx2 -axmic-avx512 -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -xcore-avx2 -axmic-avx512 -pthread" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DUSE_MKL=TRUE \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${CMBENV_AUX_ROOT}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${CMBENV_AUX_ROOT}/lib" \
    ${opts} \
    ..
