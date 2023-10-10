#!/bin/bash

# This configures toast using cmake directly (not pip) and the NVHPC
# compilers

set -e

opts="$@"

if [ "x$(which nvc++)" = "x" ]; then
    echo "The nvc++ compiler is not in your PATH"
    exit 1
fi
nvlibs=$(dirname $(dirname $(which nvc++)))/lib

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
topdir=$(dirname "${scriptdir}")

# Use the helper shell functions to load the venv path
# into the library search paths
venv_path=$(dirname $(dirname $(which python3)))

PREFIX="${venv_path}"
LIBDIR="${PREFIX}/lib"

if [ "x${DEBUG}" = "x1" ]; then
    CMAKE_BUILD_TYPE=Debug
else
    CMAKE_BUILD_TYPE=Release
fi

# Set our compiler flags
export CC=nvc
export CXX=nvc++
export FC=nvfortran
export CFLAGS="-O3 -g -fPIC -pthread -noswitcherror"
export FCFLAGS="-O3 -g -fPIC -pthread -noswitcherror"
export CXXFLAGS="-O3 -g -fPIC -pthread -std=c++11 -noswitcherror"

cmake \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="${CFLAGS}" \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DFFTW_ROOT="${PREFIX}" \
    -DAATM_ROOT="${PREFIX}" \
    -DFLAC_ROOT="${PREFIX}" \
    -DBLAS_LIBRARIES="-L${nvlibs} -lblas -lnvf -mp -lrt" \
    -DLAPACK_LIBRARIES="-L${nvlibs} -llapack -lnvf -mp -lrt" \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${PREFIX}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${LIBDIR}" \
    ${opts} ..
