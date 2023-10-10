#!/bin/bash

# This script assumes you have a python3 loaded AND you have a working
# gcc available.  Note that to build mpi4py with a custom MPI compiler,
# you should have the MPICC environment variable set to the compiler
# before running this script.

envname=$1
optional=$2

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1

# Location of build helper tools
venvpkgdir=$(dirname "${scriptdir}")/packaging/venv

if [ "x$(which g++)" = "x" ]; then
    echo "The GNU compilers are not in your PATH"
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
export CC=gcc
export CXX=g++
export FC=gfortran
export CFLAGS="-O3 -g -fPIC -pthread"
export FCFLAGS="-O3 -g -fPIC -pthread"
export CXXFLAGS="-O3 -g -fPIC -pthread -std=c++11"
export OMPFLAGS="-fopenmp"
export FCLIBS="-lgfortran"

eval "${venvpkgdir}/install_deps_venv.sh" "${envname}" ${optional}
