#!/bin/bash

# This script is used to build toast and dependencies and install them
# into the currently active virtualenv.

set -e

# Location of the source tree
pushd $(dirname $0) >/dev/null 2>&1
topdir=$(dirname $(pwd))
popd >/dev/null 2>&1

venv_path=$1
if [ "x${venv_path}" = "x" ]; then
    echo "Usage:  $0 <virtualenv path>"
    echo "  If the path to the virtualenv exists, it will be activated."
    echo "  Otherwise it will be created."
    exit 1
fi

toolchain=$2
if [ "x${toolchain}" = "x" ]; then
    toolchain="gcc"
fi

# Deactivate any current venv
if [ "x$(type -t deactivate)" != "x" ]; then
    deactivate
fi

echo "Using compiler '${toolchain}'"

if [ -d ${venv_path} ]; then
    echo "Virtualenv \"${venv_path}\" already exists, activating"
else
    echo "Creating virtualenv \"${venv_path}\""
    eval "python3 -m venv \"${venv_path}\""
fi
source "${venv_path}/bin/activate"
venv_py_ver=$(python3 --version | awk '{print $2}')

# Install our dependencies
eval "${topdir}/wheels/install_deps_linux.sh" "${toolchain}" "${venv_path}" "no"

# Look for our dependencies in the virtualenv
export LD_LIBRARY_PATH="${venv_path}/lib"
export CPATH="${venv_path}/include"

if [ "x${toolchain}" = "xgcc" ]; then
    CC=gcc
    CXX=g++
    CFLAGS="-O3 -fPIC -pthread"
    CXXFLAGS="-O3 -fPIC -pthread -std=c++11"
    FCLIBS="-lgfortran"
else
    if [ "x${toolchain}" = "xllvm" ]; then
        CC=clang-17
        CXX=clang++-17
        CFLAGS="-O3 -fPIC -pthread"
        CXXFLAGS="-O3 -fPIC -pthread -std=c++11 -stdlib=libc++"
        FCLIBS="-lgfortran"
    else
        echo "Unsupported toolchain \"${toolchain}\""
        exit 1
    fi
fi

# Set up toast build options
export TOAST_BUILD_CMAKE_C_COMPILER="${CC}"
export TOAST_BUILD_CMAKE_CXX_COMPILER="${CXX}"
export TOAST_BUILD_CMAKE_C_FLAGS="${CFLAGS} -I${venv_path}/include"
export TOAST_BUILD_CMAKE_CXX_FLAGS="${CXXFLAGS} -I${venv_path}/include"
export TOAST_BUILD_DISABLE_OPENMP_TARGET=ON
export TOAST_BUILD_BLAS_LIBRARIES="-L${venv_path}/lib -lopenblas -fopenmp -lm ${FCLIBS}"
export TOAST_BUILD_LAPACK_LIBRARIES="-L${venv_path}/lib -lopenblas -fopenmp -lm ${FCLIBS}"
export TOAST_BUILD_CMAKE_VERBOSE_MAKEFILE=ON
export TOAST_BUILD_AATM_ROOT="${venv_path}"
export TOAST_BUILD_FFTW_ROOT="${venv_path}"
export TOAST_BUILD_SUITESPARSE_INCLUDE_DIR_HINTS="${venv_path}/include"
export TOAST_BUILD_SUITESPARSE_LIBRARY_DIR_HINTS="${venv_path}/lib"
export TOAST_BUILD_CMAKE_LIBRARY_PATH="${venv_path}/lib"
# export TOAST_BUILD_TOAST_STATIC_DEPS=ON

# Install it
pushd "${topdir}" 2>&1 >/dev/null
python3 -m pip install -vvv .
popd 2>&1 >/dev/null

# Run tests
python3 -c 'import toast.tests; toast.tests.run()'
