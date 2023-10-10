#!/bin/bash

# This configures toast using cmake directly (not pip) and the NVHPC
# compilers

set -e

opts="$@"

if [ "x$(which clang++-17)" = "x" ]; then
    echo "The clang++-17 compiler is not in your PATH"
    exit 1
fi

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
INCDIR="${PREFIX}/include"

if [ "x${DEBUG}" = "x1" ]; then
    CMAKE_BUILD_TYPE=Debug
else
    CMAKE_BUILD_TYPE=Release
fi

# Set our compiler flags
export CC=clang-17
export CXX=clang++-17
export FC=gfortran
export CFLAGS="-O3 -g -fPIC -pthread"
export FCFLAGS="-O3 -g -fPIC -pthread"
export CXXFLAGS="-O3 -g -fPIC -pthread -std=c++11 -stdlib=libc++ -I${INCDIR}"

export LD_LIBRARY_PATH="/usr/lib/llvm-17/lib:/usr/lib/gcc/x86_64-linux-gnu/11:${LIBDIR}"
export LIBRARY_PATH="/usr/lib/llvm-17/lib:/usr/lib/gcc/x86_64-linux-gnu/11:${LIBDIR}"

# # export CMAKE_MODULE_LINKER_FLAGS="-L/usr/lib/llvm-17/lib -stdlib=libc++ "
# export LINK_OPTIONS="-stdlib=libc++"

cmake \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="${CFLAGS}" \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    -DUSE_OPENMP_TARGET=TRUE \
    -DOPENMP_TARGET_FLAGS="-fopenmp -fopenmp-targets=nvptx64 -fopenmp-target-debug=3 --offload-arch=sm_86 -Wl,-lomp,-lomptarget,-lomptarget.rtl.cuda" \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DFFTW_ROOT="${PREFIX}" \
    -DAATM_ROOT="${PREFIX}" \
    -DFLAC_ROOT="${PREFIX}" \
    -DBLAS_LIBRARIES="${LIBDIR}/libopenblas.so;/usr/lib/gcc/x86_64-linux-gnu/11/libgfortran.so" \
    -DLAPACK_LIBRARIES="${LIBDIR}/libopenblas.so;/usr/lib/gcc/x86_64-linux-gnu/11/libgfortran.so" \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${PREFIX}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${LIBDIR}" \
    ${opts} ..
