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
# $> ./cmake/platforms/osx_macports.sh --with_elemental=$HOME/software \
#    -DCMAKE_PREFIX_PATH=$HOME/software
#

ROOT=${PWD}
DIR=${PWD}/build-toast/osx-macports/release
OPTS="$@"

export CC=$(which gcc)
export CXX=$(which g++)
export MPICC=$(which gcc)
export MPICXX=$(which g++)

# ${OPTS} at end because last -D overrides any previous -D with same name
# on command line (e.g. -DUSE_MPI=OFF -DUSE_MPI=OFF --> USE_MPI == OFF)
mkdir -p ${DIR} 
cd ${DIR}
cmake -DCMAKE_BUILD_TYPE=Release \
    -DUSE_MKL=OFF -DUSE_TBB=OFF -DFFTW_ROOT=/opt/local \
    ${OPTS} ${ROOT}

