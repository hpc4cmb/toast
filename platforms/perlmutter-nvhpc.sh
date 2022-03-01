#!/bin/bash

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="nvc" \
    -DCMAKE_CXX_COMPILER="nvc++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread -std=c++11" \
    -DMKL_DISABLED=TRUE \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DAATM_ROOT="${CMBENV_AUX_ROOT}" \
    -DFFTW_ROOT="${CMBENV_AUX_ROOT}" \
    -DBLAS_LIBRARIES=${CMBENV_AUX_ROOT}/lib/libopenblas.so \
    -DLAPACK_LIBRARIES=${CMBENV_AUX_ROOT}/lib/libopenblas.so \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${CMBENV_AUX_ROOT}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${CMBENV_AUX_ROOT}/lib" \
    ${opts} \
    ..

