#!/bin/bash

# Pass extra configure options to this script, including
# things like --prefix, --with-elemental, etc.

OPTS="$@"

export CC=mpicc
export CXX=mpicxx
export MPICC=mpicc
export MPICXX=mpicxx
export CFLAGS="-O3 -g -fPIC -pthread"
export CXXFLAGS="-O3 -g -fPIC -pthread"
export OPENMP_CFLAGS="-fopenmp"
export OPENMP_CXXFLAGS="-fopenmp"
export LDFLAGS="-lpthread"

./configure ${OPTS} 
