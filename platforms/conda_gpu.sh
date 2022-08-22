#!/bin/bash

# This is similar to the "conda_dev.sh" script, but uses a target-offload version of
# gcc.  However, it also requires a manual hack- replacing the libgomp library in the
# conda env with a symlink to the nvptx enabled version that comes with the custom gcc.

set -e

if [ "x${CONDA_PREFIX}" = "x" ]; then
    echo "You must activate a conda environment before using this script."
    exit 1
fi

# if [ "x${CONDA_TOOLCHAIN_HOST}" = "x" ]; then
#     echo "Your conda environment does not contain compilers.  Run conda_dev_setup.sh first."
#     exit 1
# fi

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
topdir=$(dirname "${scriptdir}")

PREFIX="${CONDA_PREFIX}"
PYTHON="${PREFIX}/bin/python"
LIBDIR="${PREFIX}/lib"

if [ "x${DEBUG}" = "x1" ]; then
    CMAKE_BUILD_TYPE=Debug
else
    CMAKE_BUILD_TYPE=Release
fi

CMAKE_PLATFORM_FLAGS=""
shext="so"
if [[ ${CONDA_TOOLCHAIN_HOST} =~ .*darwin.* ]]; then
    CMAKE_PLATFORM_FLAGS+=(-DCMAKE_OSX_SYSROOT="${CONDA_BUILD_SYSROOT}")
    shext="dylib"
fi

cmake \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" ${CMAKE_PLATFORM_FLAGS} \
    -DCMAKE_C_COMPILER="gcc-12" \
    -DCMAKE_CXX_COMPILER="g++-12" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread -std=c++11 -fcf-protection=none -fno-stack-protector" \
    -DUSE_OPENMP_TARGET=TRUE \
    -DOPENMP_TARGET_FLAGS="-foffload=nvptx-none -foffload-options=-Wa,-m,sm_80 -foffload-options=-misa=sm_80 -foffload-options=-fPIC -foffload-options=-lm -foffload-options=-latomic" \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DFFTW_ROOT="${PREFIX}" \
    -DAATM_ROOT="${PREFIX}" \
    -DBLAS_LIBRARIES="${LIBDIR}/libblas.${shext}" \
    -DLAPACK_LIBRARIES="${LIBDIR}/liblapack.${shext}" \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${PREFIX}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${LIBDIR}" \
    ..
