#!/bin/bash

# This script assumes you have a python3 loaded AND an installation
# of LLVM with OpenMP target offload support.  If the second option
# to this script is "yes", you must set the MPICC variable to your
# MPI compiler for building mpi4py.

envname=$1
optional=$2

if [ "x${optional}" != "xyes" ]; then
    optional=no
fi

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1

# Location of build helper tools
venvpkgdir=$(dirname "${scriptdir}")/packaging/venv

suf="-18"
if [ "x$(which clang++${suf})" = "x" ]; then
    echo "The clang++${suf} compiler is not in your PATH, trying clang++"
    if [ "x$(which clang++)" = "x" ]; then
        echo "No clang++ found"
        exit 1
    else
        suf=""
    fi
fi

usage () {
    echo ""
    echo "Usage:  $0 <path to virtualenv>"
    echo ""
    echo "The virtualenv will be created and / or activated"
    echo ""
}

if [ "x${envname}" = "x" ]; then
    usage
    exit 1
fi

# Set our compiler flags
export CC=clang${suf}
export CXX=clang++${suf}
export FC=gfortran
export CFLAGS="-O3 -g -fPIC -pthread"
export FCFLAGS="-O3 -g -fPIC -pthread"
export CXXFLAGS="-O3 -g -fPIC -pthread -std=c++11 -stdlib=libc++"
export OMPFLAGS="-fopenmp"
export FCLIBS="-L/usr/lib/gcc/x86_64-linux-gnu/11 -lgfortran"

export LD_LIBRARY_PATH="/usr/lib/llvm-18/lib:/usr/lib/gcc/x86_64-linux-gnu/11:${envname}/lib"

eval "${venvpkgdir}/install_deps_venv.sh" "${envname}" ${optional} yes
