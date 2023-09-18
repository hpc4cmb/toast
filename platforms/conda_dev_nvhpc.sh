#!/bin/bash

# This configures toast using cmake directly (not pip), but uses the
# conda compilers and other dependencies provided by conda.  It is
# useful for building toast in parallel repeatedly for debugging
# in situations where clean un-install is not a concern.

set -e

opts="$@"

if [ "x${CONDA_PREFIX}" = "x" ]; then
    echo "You must activate a conda environment before using this script."
    exit 1
fi

if [ "x$(which nvc++)" = "x" ]; then
    echo "The nvc++ compiler is not in your PATH"
    exit 1
fi

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
topdir=$(dirname "${scriptdir}")

# Use the helper shell functions to load the sidecar
# directory of compiled packages.
ext_path="${CONDA_PREFIX}_ext"
if [ ! -d "${ext_path}" ]; then
    echo "External package directory '${ext_path}' does not exist."
    echo "Did you use conda_dev_setup_nvhpc.sh to create the env?"
    exit 1
fi
source "${topdir}/packaging/conda/load_conda_external.sh"
prepend_ext_env "PATH" "${ext_path}/bin"
prepend_ext_env "CPATH" "${ext_path}/include"
prepend_ext_env "LIBRARY_PATH" "${ext_path}/lib"
prepend_ext_env "LD_LIBRARY_PATH" "${ext_path}/lib"
prepend_ext_env "PKG_CONFIG_PATH" "${ext_path}/lib/pkgconfig"

PREFIX="${ext_path}"
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
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
    -DFFTW_ROOT="${PREFIX}" \
    -DAATM_ROOT="${PREFIX}" \
    -DFLAC_ROOT="${PREFIX}" \
    -DBLAS_LIBRARIES="${LIBDIR}/libopenblas.so" \
    -DLAPACK_LIBRARIES="${LIBDIR}/libopenblas.so" \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${PREFIX}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${LIBDIR}" \
    ${opts} ..
