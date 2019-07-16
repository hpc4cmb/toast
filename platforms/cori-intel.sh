#!/bin/bash

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="${CRAYPE_DIR}/bin/cc" \
    -DCMAKE_CXX_COMPILER="${CRAYPE_DIR}/bin/CC" \
    -DMPI_C_COMPILER="${CRAYPE_DIR}/bin/cc" \
    -DMPI_CXX_COMPILER="${CRAYPE_DIR}/bin/CC" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -xcore-avx2 -axmic-avx512 -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -xcore-avx2 -axmic-avx512 -pthread -std=c++11" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DBLAS_LIBRARIES=$MKLROOT/lib/intel64/libmkl_rt.so \
    -DLAPACK_LIBRARIES=$MKLROOT/lib/intel64/libmkl_rt.so \
    -DFFTW_ROOT=$CMBENV_AUX_ROOT \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${CMBENV_AUX_ROOT}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${CMBENV_AUX_ROOT}/lib" \
    ${opts} \
    ..


