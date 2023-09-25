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

source "${topdir}/packaging/conda/load_conda_external.sh"
prepend_ext_env "PATH" "${venv_path}/bin"
prepend_ext_env "CPATH" "${venv_path}/include"
prepend_ext_env "LIBRARY_PATH" "${venv_path}/lib"
prepend_ext_env "LD_LIBRARY_PATH" "${venv_path}/lib"
prepend_ext_env "PKG_CONFIG_PATH" "${venv_path}/lib/pkgconfig"

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
export CFLAGS="-O3 -g -fPIC -pthread"
export FCFLAGS="-O3 -g -fPIC -pthread"
export CXXFLAGS="-O3 -g -fPIC -pthread -std=c++11"
export OMPFLAGS="-fopenmp"

cmake \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="nvc" \
    -DCMAKE_CXX_COMPILER="nvc++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread -std=c++11" \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DFFTW_ROOT="${PREFIX}" \
    -DAATM_ROOT="${PREFIX}" \
    -DFLAC_ROOT="${PREFIX}" \
    -DBLAS_LIBRARIES="${nvlibs}/libblas.so;${nvlibs}/libnvf.so" \
    -DLAPACK_LIBRARIES="${nvlibs}/liblapack.so;${nvlibs}/libnvf.so" \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${PREFIX}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${LIBDIR}" \
    ${opts} ..
