#!/bin/bash

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="${CRAYPE_DIR}/bin/cc" \
    -DCMAKE_CXX_COMPILER="${CRAYPE_DIR}/bin/CC" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DBLAS_LIBRARIES=$CMBENV_AUX_ROOT/lib/libopenblas.a \
    -DLAPACK_LIBRARIES=$CMBENV_AUX_ROOT/lib/libopenblas.a \
    -DFFTW_ROOT=$CMBENV_AUX_ROOT \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${CMBENV_AUX_ROOT}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${CMBENV_AUX_ROOT}/lib" \
    ${opts} \
    ..
