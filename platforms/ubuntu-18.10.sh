#!/bin/bash

# Pass extra configure options to this script, including
# things like --prefix, --with-elemental, etc.

OPTS="$@"

export PYTHON=python3
export CC=mpicc
export CXX=mpicxx
export MPICC=mpicc
export MPICXX=mpicxx
export CFLAGS="-O3 -g -fPIC -pthread -std=c99"
export CXXFLAGS="-O3 -g -fPIC -pthread -std=c++11 -fext-numeric-literals"
export OPENMP_CFLAGS="-fopenmp"
export OPENMP_CXXFLAGS="-fopenmp"
export LDFLAGS="-lpthread"
export CPPFLAGS="-I/usr/include/suitesparse"
./configure ${OPTS} --with-suitesparse=/usr
