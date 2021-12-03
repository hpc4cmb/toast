#!/bin/bash

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="nvc" \
    -DCMAKE_CXX_COMPILER="nvc++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread -std=c++11" \
    -DMKL_DISABLED=TRUE \
    -DUSE_CUDA=TRUE \
    -DUSE_OPENACC=TRUE \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DBLAS_LIBRARIES="-lopenblas -lgomp -lm" \
    ${opts} \
    ..

