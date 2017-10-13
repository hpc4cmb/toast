#!/bin/bash

# This script assumes that the "toast-deps" module has been loaded.
# To install to my scratch directory, I might run this script with:
#
# $> ./platforms/cori-gcc_mkl.sh \
#    --prefix=$SCRATCH/software/toast-gcc
#

OPTS="$@"
WARN_FLAGS="-W -Wall -Wno-deprecated -Wwrite-strings -Wpointer-arith"
WARN_FLAGS="${WARN_FLAGS} -pedantic -Wshadow -Wextra -Wno-unused-parameter"
C_WARN_FLAGS="${WARN_FLAGS}"
CXX_WARN_FLAGS=${WARN_FLAGS} -Woverloaded-virtual"

export CC=cc
export CXX=CC
export MPICC=cc
export MPICXX=CC
export CFLAGS="-O3 -g ${C_WARN_FLAGS} -fPIC"
export CXXFLAGS="-O3 -g ${CXX_WARN_FLAGS} -fPIC"
export OPENMP_CFLAGS="-fopenmp"
export OPENMP_CXXFLAGS="-fopenmp"

./configure ${OPTS} \
    --without-mkl \
    --with-blas="-lopenblas -fopenmp"

