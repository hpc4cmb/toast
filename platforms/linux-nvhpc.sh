#!/bin/bash

# Pass extra cmake options to this script, including
# things like -DCMAKE_INSTALL_PREFIX=/path/to/install, etc.

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="nvc" \
    -DCMAKE_CXX_COMPILER="nvc++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread -std=c++11" \
    -DMKL_DISABLED=TRUE \
    -DUSE_OPENACC=TRUE \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DBLAS_LIBRARIES="-L${NVHPC_ROOT}/math_libs/lib64 -lblas -lnvblas -lnvf -lm" \
    -DLAPACK_LIBRARIES="-llapack" \
    ${opts} \
    ..

# FIXME: remove fftw once cufft integrated
