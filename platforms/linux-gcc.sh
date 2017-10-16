#!/bin/bash

# Pass extra configure options to this script, including
# things like --prefix, --with-elemental, etc.

set -o errexit

OPTS="$@"

# get flags from flags/gcc.sh
. $(dirname ${BASH_SOURCE[0]})/flags/gcc.sh

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
