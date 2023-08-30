#!/bin/bash
#
# This script is designed to run within a container managed by cibuildwheel.
# This will run in a manylinux2014 (CentOS 7) container.
#
# The purpose of this script is to install TOAST dependency libraries that will be
# bundled with our compiled extension.
#

set -e

toolchain=$1
prefix=$2
static=$3

if [ "x${toolchain}" = "x" ]; then
    toolchain="gcc"
fi

if [ "x${prefix}" = "x" ]; then
    prefix=/usr/local
fi

if [ "x${static}" = "x" ]; then
    static="yes"
fi

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
echo "Wheel script directory = ${scriptdir}"

# Location of dependency scripts
depsdir=$(dirname ${scriptdir})/deps

# Build options

if [ "x${toolchain}" = "xgcc" ]; then
    export CC=gcc
    export CXX=g++
    export FC=gfortran

    export CFLAGS="-O3 -fPIC -pthread"
    export FCFLAGS="-O3 -fPIC -pthread"
    export CXXFLAGS="-O3 -fPIC -pthread -std=c++11"
    export FCLIBS="-lgfortran"
    export OMPFLAGS="-fopenmp"
else
    if [ "x${toolchain}" = "xllvm" ]; then
        export CC=clang-17
        export CXX=clang++-17
        export FC=gfortran

        export CFLAGS="-O3 -fPIC -pthread"
        export FCFLAGS="-O3 -fPIC -pthread"
        export CXXFLAGS="-O3 -fPIC -pthread -std=c++11 -stdlib=libc++"
        export FCLIBS="-L/usr/lib/llvm-17/lib /usr/lib/x86_64-linux-gnu/libgfortran.so.5"
        export OMPFLAGS="-fopenmp"
    else
        echo "Unsupported toolchain \"${toolchain}\""
        exit 1
    fi
fi

# Update pip
pip install --upgrade pip

# Install a couple of base packages that are always required
pip install -v cmake wheel

# In order to maximize ABI compatibility with numpy, build with the newest numpy
# version containing the oldest ABI version compatible with the python we are using.
pyver=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
if [ ${pyver} == "3.8" ]; then
    numpy_ver="1.20"
fi
if [ ${pyver} == "3.9" ]; then
    numpy_ver="1.20"
fi
if [ ${pyver} == "3.10" ]; then
    numpy_ver="1.22"
fi
if [ ${pyver} == "3.11" ]; then
    numpy_ver="1.24"
fi

# Install build requirements.
CC="${CC}" CFLAGS="${CFLAGS}" pip install -v "numpy<${numpy_ver}" -r "${scriptdir}/build_requirements.txt"

# Build compiled dependencies

# For testing locally
# export MAKEJ=8
export MAKEJ=2
export PREFIX="${prefix}"
export DEPSDIR="${depsdir}"
export STATIC="${static}"
export SHLIBEXT="so"
export CLEANUP=yes

for pkg in openblas fftw libflac suitesparse libaatm; do
    source "${depsdir}/${pkg}.sh"
done
