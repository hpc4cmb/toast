#!/bin/bash

# Pass extra configure options to this script, including
# things like --prefix, --with-elemental, etc.

OPTS="$@"
WARN_FLAGS="-W -Wall -Wno-deprecated -Wwrite-strings -Wpointer-arith"
WARN_FLAGS="${WARN_FLAGS} -pedantic -Wshadow -Wextra -Wno-unused-parameter"
C_WARN_FLAGS="${WARN_FLAGS}"
CXX_WARN_FLAGS=${WARN_FLAGS} -Woverloaded-virtual"

export CC=mpicc
export CXX=mpicxx
export MPICC=mpicc
export MPICXX=mpicxx
export CFLAGS="-O3 -g ${C_WARN_FLAGS} -fPIC -pthread"
export CXXFLAGS="-O3 -g ${CXX_WARN_FLAGS} -fPIC -pthread"
export OPENMP_CFLAGS="-fopenmp"
export OPENMP_CXXFLAGS="-fopenmp"
export LDFLAGS="-lpthread"

./configure ${OPTS} 
