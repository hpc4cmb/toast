#!/bin/bash

# Pass extra cmake options to this script, including
# things like -DCMAKE_INSTALL_PREFIX=/path/to/install, etc.

opts="$@"

if [ "x${CMBENV_AUX_ROOT}" = "x" ]; then
    echo "You must activate a cmbenv environment before using this script."
    exit 1
fi

if [ "x${DEBUG}" = "x1" ]; then
    CMAKE_BUILD_TYPE=Debug
    oflags="-O0"
else
    CMAKE_BUILD_TYPE=Release
    oflags="-O3"
fi

cmake \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="nvc" \
    -DCMAKE_CXX_COMPILER="nvc++" \
    -DCMAKE_C_FLAGS="${oflags} -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="${oflags} -g -fPIC -pthread -std=c++11" \
    -DOPENMP_TARGET_FLAGS="-Minfo=mp -gpu=cc86" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DUSE_OPENMP_TARGET=TRUE \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DCMAKE_INSTALL_PREFIX="${CMBENV_AUX_ROOT}" \
    -DFFTW_ROOT="${CMBENV_AUX_ROOT}" \
    -DDISABLE_FLAC=1 \
    -DBLAS_LIBRARIES="-L${NVHPC_ROOT}/compilers/lib -lblas -lnvf -mp" \
    -DLAPACK_LIBRARIES="-L${NVHPC_ROOT}/compilers/lib -llapack -lnvf -mp" \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${CMBENV_AUX_ROOT}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${CMBENV_AUX_ROOT}/lib" ${opts} \
    ..

