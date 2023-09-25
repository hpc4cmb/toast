#!/bin/bash

# This script assumes you have a python3 loaded AND
# you have the NVHPC SDK loaded (through a modulefile or similar).
# Note that to build mpi4py with a custom MPI compiler, you should
# load the nvhpc-nompi module and set the MPICC environment variable
# before running this script.  Otherwise load the normal nvhpc module
# and use the MPI that ships with NVHPC.

envname=$1
optional=$2

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1

# Location of build helper tools
venvpkgdir=$(dirname "${scriptdir}")/packaging/venv

if [ "x$(which nvc++)" = "x" ]; then
    echo "The nvc++ compiler is not in your PATH"
    exit 1
fi
nvlibs=$(dirname $(dirname $(which nvc++)))/lib

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
export CC=nvc
export CXX=nvc++
export FC=nvfortran
export CFLAGS="-O3 -g -fPIC -pthread"
export FCFLAGS="-O3 -g -fPIC -pthread"
export CXXFLAGS="-O3 -g -fPIC -pthread -std=c++11"
export OMPFLAGS="-fopenmp"
export FCLIBS="-lnvf -lrt"
export BLAS_LIBRARIES="-L${nvlibs} -lblas -lnvf -mp -lrt"
export LAPACK_LIBRARIES="-L${nvlibs} -llapack -lnvf -mp -lrt"
export MPICC="$(which mpicc) -noswitcherror"

eval "${venvpkgdir}/install_deps_venv.sh" "${envname}" ${optional}
