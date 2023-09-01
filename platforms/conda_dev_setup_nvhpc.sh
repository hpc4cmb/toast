#!/bin/bash

# This script assumes you have a conda base environment loaded AND
# you have the NVHPC SDK loaded (through a modulefile or similar).
# Note that to build mpi4py with a custom MPI compiler, you should
# load the nvhpc-nompi module and set the MPICC environment variable
# before running this script.

envname=$1
pyversion=$2
optional=$3

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1

# Location of build helper tools
condapkgdir=$(dirname "${scriptdir}")/packaging/conda

conda_exe=$(which conda)
if [ "x${conda_exe}" = "x" ]; then
    echo "No conda executable found"
    exit 1
fi

if [ "x$(which nvc++)" = "x" ]; then
    echo "The nvc++ compiler is not in your PATH"
    exit 1
fi

usage () {
    echo ""
    echo "Usage:  $0 <name of conda env or path>"
    echo ""
    echo "The named environment will be activated (and created if needed)"
    echo ""
}

if [ "x${envname}" = "x" ]; then
    usage
    exit 1
fi

# Set our compiler flags
export CC=nvc
export CXX=nvc++
export FC=nvfortran
export CFLAGS="-O3 -g -fPIC -pthread"
export FCFLAGS="-O3 -g -fPIC -pthread"
export CXXFLAGS="-O3 -g -fPIC -pthread -std=c++11"
export OMPFLAGS="-fopenmp"

eval "${condapkgdir}/install_deps_conda_external.sh" "${envname}" ${pyversion} ${optional}
