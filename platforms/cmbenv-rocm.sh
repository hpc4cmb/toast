#!/bin/bash

# Pass extra cmake options to this script, including
# things like -DCMAKE_INSTALL_PREFIX=/path/to/install, etc.

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="amdclang" \
    -DCMAKE_CXX_COMPILER="amdclang++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC" \
    -DUSE_OPENMP_TARGET=TRUE \
    -DOPENMP_TARGET_FLAGS="\
    -fopenmp-target-debug=3 \
    -fopenmp-targets=amdgcn-amd-amdhsa \
    -Xopenmp-target=amdgcn-amd-amdhsa \
    -march=gfx1030 \
    " \
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
