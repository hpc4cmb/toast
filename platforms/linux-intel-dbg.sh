#!/bin/bash

# Pass extra configure options to this script, including
# things like --prefix, --with-elemental, etc.

OPTS="$@"

export PYTHON=python3
export CC=mpiicc
export CXX=mpiicpc
export MPICC=mpiicc
export MPICXX=mpiicpc
export CFLAGS="-O0 -g -fPIC -pthread"
export CXXFLAGS="-O0 -g -fPIC -pthread"
export OPENMP_CFLAGS="-qopenmp"
export OPENMP_CXXFLAGS="-qopenmp"

./configure ${OPTS} \
    --with-math="-limf -lsvml" \
    --with-mkl="${MKLROOT}/lib/intel64" \
