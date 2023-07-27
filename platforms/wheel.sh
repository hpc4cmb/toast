#!/bin/bash

set -e

PYTHON=$(which python3)
PREFIX=$(dirname $(dirname "${PYTHON}"))
LIBDIR="${PREFIX}/lib"

echo "Using Python '${PYTHON}'"
echo "Installing to '${PREFIX}'"

os=linux
if [ "x$(uname)" != "xLinux" ]; then
    os=darwin
fi

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
topdir=$(dirname "${scriptdir}")

if [ "x${DEBUG}" = "x1" ]; then
    CMAKE_BUILD_TYPE=Debug
else
    CMAKE_BUILD_TYPE=Release
fi

shext="so"
if [ "${os}" = "darwin" ]; then
    shext="dylib"
fi

if [ "x${CC}" = "x" ]; then
    echo "Set the CC variable to desired C compiler"
    exit 1
fi

if [ "x${CXX}" = "x" ]; then
    echo "Set the CXX variable to desired C++ compiler"
    exit 1
fi

export TOAST_BUILD_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
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

pushd "${topdir}" >/dev/null 2>&1

# Now build a wheel
rm -rf build
python3 setup.py clean
python3 -m pip wheel --wheel-dir=build/temp_wheels --no-deps -vvv .

input_wheel=$(ls ${topdir}/build/temp_wheels/*.whl)
wheel_file=$(basename ${input_wheel})

# Repair it
if [ "${os}" = "darwin" ]; then
    delocate-listdeps ${input_wheel} \
    && delocate-wheel -w "${topdir}/wheelhouse" ${input_wheel}
else
    auditwheel show ${input_wheel} \
    && auditwheel repair ${input_wheel}
fi

# Install it
python3 -m pip install "${topdir}/wheelhouse/${wheel_file}"

popd >/dev/null 2>&1
