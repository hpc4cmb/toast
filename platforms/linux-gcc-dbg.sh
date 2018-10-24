#!/bin/bash

# Pass extra configure options to this script, including
# things like --prefix, --with-elemental, etc.

OPTS="$@"

export PYTHON=python3
export CC=mpicc
export CXX=mpicxx
export MPICC=mpicc
export MPICXX=mpicxx
export CFLAGS="-O0 -g -fPIC -pthread -std=c99"
export CXXFLAGS="-O0 -g -fPIC -pthread -std=c++11 -fext-numeric-literals"
export OPENMP_CFLAGS="-fopenmp"
export OPENMP_CXXFLAGS="-fopenmp"
export LDFLAGS="-lpthread"

./configure ${OPTS}
