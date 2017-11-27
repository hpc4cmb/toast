#!/bin/bash

# Pass extra configure options to this script, including
# things like --prefix, --with-elemental, etc.

OPTS="$@"

export CC=mpiicc
export CXX=mpiicpc
export MPICC=mpiicc
export MPICXX=mpiicpc
export CFLAGS="-O3 -g -fPIC -pthread"
export CXXFLAGS="-O3 -g -fPIC -pthread"
export OPENMP_CFLAGS="-qopenmp"
export OPENMP_CXXFLAGS="-qopenmp"

./configure ${OPTS} \
    --with-math="-limf -lsvml" \
    --with-mkl="${MKLROOT}/lib/intel64" \
    --with-tbb="${TBBROOT}/lib/intel64/gcc4.7"
