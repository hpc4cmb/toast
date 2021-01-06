#!/bin/bash

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="gcc" \
    -DCMAKE_CXX_COMPILER="g++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DMKL_DISABLED=TRUE \
    -DBLAS_LIBRARIES=$CMBENV_AUX_ROOT/lib/libopenblas.so \
    -DLAPACK_LIBRARIES=$CMBENV_AUX_ROOT/lib/libopenblas.so \
    -DFFTW_ROOT=$CMBENV_AUX_ROOT \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${CMBENV_AUX_ROOT}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${CMBENV_AUX_ROOT}/lib" \
    ${opts} \
    ..
