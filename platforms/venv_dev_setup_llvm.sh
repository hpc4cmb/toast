#!/bin/bash

# This script assumes you have a python3 loaded AND an installation
# of LLVM with OpenMP target offload support.  If the second option
# to this script is "yes", you must set the MPICC variable to your
# MPI compiler for building mpi4py.

envname=$1
optional=$2

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1

# Location of build helper tools
venvpkgdir=$(dirname "${scriptdir}")/packaging/venv

if [ "x$(which clang++-17)" = "x" ]; then
    echo "The clang++-17 compiler is not in your PATH"
    exit 1
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
export CC=clang-17
export CXX=clang++-17
export FC=flang-new-17
export CFLAGS="-O3 -g -fPIC -pthread"
export FCFLAGS="-O3 -g -fPIC -pthread"
export CXXFLAGS="-O3 -g -fPIC -pthread -std=c++11 -stdlib=libc++"
export OMPFLAGS="-fopenmp"
export FCLIBS=""

export LD_LIBRARY_PATH="/usr/lib/llvm-17/lib:${envname}/lib"

eval "${venvpkgdir}/install_deps_venv.sh" "${envname}" ${optional}
