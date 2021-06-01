#!/bin/bash

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="gcc" \
    -DCMAKE_CXX_COMPILER="g++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_SHARED_LINKER_FLAGS="-lgfortran" \
    -DCMAKE_EXE_LINKER_FLAGS="-lgfortran" \
    -DCMAKE_MODULE_LINKER_FLAGS="-lgfortran" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DMKL_DISABLED=TRUE \
    -DTOAST_STATIC_DEPS:BOOL=ON \
    -DFFTW_ROOT=$CMBENV_AUX_ROOT \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${CMBENV_AUX_ROOT}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${CMBENV_AUX_ROOT}/lib" \
    ${opts} \
    ..

