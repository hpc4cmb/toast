#!/bin/bash

# Pass extra cmake options to this script, including
# things like -DCMAKE_INSTALL_PREFIX=/path/to/install, etc.

opts="$@"

# NOTE:  You can toggle use of the mempool with OpenACC by
# passing:
#    -DUSE_OPENACC_MEMPOOL=TRUE
# to this script.

cmake \
    -DCMAKE_C_COMPILER="nvc" \
    -DCMAKE_CXX_COMPILER="nvc++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread -std=c++11" \
    -DMKL_DISABLED=TRUE \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DBLAS_LIBRARIES="-lopenblas -lgomp -lm" \
    ${opts} \
    ..
