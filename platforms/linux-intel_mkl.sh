#!/bin/bash

OPTS="$@"

export CC=mpiicc
export CXX=mpiicpc
export MPICC=mpiicc
export MPICXX=mpiicpc
export CFLAGS="-O3 -g -fPIC -pthread"
export CXXFLAGS="-O3 -g -fPIC -pthread"
export OPENMP_CFLAGS="-qopenmp"
export OPENMP_CXXFLAGS="-qopenmp"
export LDFLAGS="-lpthread -liomp5"

./configure ${OPTS} \
    --with-math="-limf" \
    --with-mkl="${MKLROOT}/lib/intel64" \
    --with-elemental="${HOME}/software/intel/idpdev"
