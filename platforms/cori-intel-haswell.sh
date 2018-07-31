#!/bin/bash

# This script assumes that the "toast-deps" module has been loaded.
# To install to my scratch directory, I might run this script with:
#
# $> ./platforms/cori-haswell_intel_mkl.sh \
#    --prefix=$SCRATCH/software/toast-haswell
#

OPTS="$@"

export PYTHON=python3
export CC=cc
export CXX=CC
export MPICC=cc
export MPICXX=CC
export CFLAGS="-O3 -g -fPIC -xcore-avx2 -pthread"
export CXXFLAGS="-O3 -g -fPIC -xcore-avx2 -pthread"
export OPENMP_CFLAGS="-qopenmp"
export OPENMP_CXXFLAGS="-qopenmp"

./configure ${OPTS} \
    --with-math="-limf -lsvml" \
    --with-mkl="${INTEL_PATH}/linux/mkl/lib/intel64"
