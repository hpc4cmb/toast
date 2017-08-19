#!/bin/bash

# This script assumes that the "toast-deps" module has been loaded.
# To install to my scratch directory, I might run this script with:
#
# $> ./platforms/cori-gcc_mkl.sh \
#    --prefix=$SCRATCH/software/toast-gcc
#

OPTS="$@"

export CC=cc
export CXX=CC
export MPICC=cc
export MPICXX=CC
export CFLAGS="-O3 -g -fPIC"
export CXXFLAGS="-O3 -g -fPIC"
export OPENMP_CFLAGS="-fopenmp"
export OPENMP_CXXFLAGS="-fopenmp"

./configure ${OPTS} \
    --without-mkl \
    --with-blas="-lopenblas -fopenmp"

