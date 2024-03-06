#!/bin/bash

set -e

INSTALL=$1

if [ "x${CONDA_PREFIX}" = "x" ]; then
    echo "You must activate a conda environment before using this script."
    exit 1
fi

if [ "x${CONDA_TOOLCHAIN_HOST}" = "x" ]; then
    echo "Your conda environment does not contain compilers.  Run conda_dev_setup.sh first."
    exit 1
fi

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

shext="so"
if [[ ${CONDA_TOOLCHAIN_HOST} =~ .*darwin.* ]]; then
    export TOAST_BUILD_CMAKE_OSX_SYSROOT="${CONDA_BUILD_SYSROOT}"
    shext="dylib"
fi

export TOAST_BUILD_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
export TOAST_BUILD_CMAKE_PLATFORM_FLAGS=${CMAKE_PLATFORM_FLAGS}
export TOAST_BUILD_CMAKE_C_COMPILER="${CC}"
export TOAST_BUILD_CMAKE_CXX_COMPILER="${CXX}"
export TOAST_BUILD_CMAKE_C_FLAGS="-O3 -g -fPIC"
export TOAST_BUILD_CMAKE_CXX_FLAGS="-O3 -g -fPIC"
export TOAST_BUILD_CMAKE_VERBOSE_MAKEFILE=1
export TOAST_BUILD_CMAKE_INSTALL_PREFIX="${PREFIX}"
export TOAST_BUILD_CMAKE_PREFIX_PATH="${PREFIX}"
export TOAST_BUILD_FFTW_ROOT="${PREFIX}"
export TOAST_BUILD_AATM_ROOT="${PREFIX}"
export TOAST_BUILD_BLAS_LIBRARIES="${LIBDIR}/libblas.${shext}"
export TOAST_BUILD_LAPACK_LIBRARIES="${LIBDIR}/liblapack.${shext}"
export TOAST_BUILD_SUITESPARSE_INCLUDE_DIR_HINTS="${PREFIX}/include"
export TOAST_BUILD_SUITESPARSE_LIBRARY_DIR_HINTS="${LIBDIR}"

# Ensure that stale build products are removed
rm -rf "${topdir}/build"

if [ -z "${INSTALL}" ]; then
    pip install -vvv .
else
    pip install -vvv --prefix "${INSTALL}" .
fi
