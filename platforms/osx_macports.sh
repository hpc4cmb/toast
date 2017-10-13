#!/bin/bash

# This script assumes that you have installed fftw, python3, gcc and MPI
# wrappers from macports.  You may also have installed Elemental
# separately as there is no macports package yet.
#
# Note that on OSX, Elemental uses relative paths to its dependencies.
# You can only run the toast_test executable after make_install:
# "make check" will fail.
#
# You can supply extra arguments to the configure script from the
# command line:
#
# $> ./platforms/osx_macports.sh --with_elemental=$HOME/software \
#    --prefix=$HOME/software
#

OPTS="$@"
WARN_FLAGS="-W -Wall -Wno-deprecated -Wwrite-strings -Wpointer-arith"
WARN_FLAGS="${WARN_FLAGS} -pedantic -Wshadow -Wextra -Wno-unused-parameter"
C_WARN_FLAGS="${WARN_FLAGS}"
CXX_WARN_FLAGS=${WARN_FLAGS} -Woverloaded-virtual"

export PYTHON=python3
export CC=mpicc
export CXX=mpicxx
export MPICC=mpicc
export MPICXX=mpicxx
export CFLAGS="-O3 -std=c99 ${C_WARN_FLAGS}"
export CXXFLAGS="-O3 ${CXX_WARN_FLAGS}"
export LDFLAGS="-Wl,-twolevel_namespace"
export PYTHON_EXTRA_LDFLAGS=$(python3-config --ldflags)

./configure ${OPTS} \
	    --with-fftw=/opt/local
