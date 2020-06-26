#!/bin/bash

# Pass extra cmake options to this script, including
# things like -DCMAKE_INSTALL_PREFIX=/path/to/install, etc.

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="icc" \
    -DCMAKE_CXX_COMPILER="icpc" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread" \
    -DBLAS_LIBRARIES=${MKLROOT}/lib/intel64/libmkl_rt.so \
    -DLAPACK_LIBRARIES=${MKLROOT}/lib/intel64/libmkl_rt.so \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    ${opts} \
    ..
