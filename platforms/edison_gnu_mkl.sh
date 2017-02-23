#!/bin/bash

# This script assumes that the "toast-deps" module has been loaded and that
# you have a version of elemental installed somewhere (Elemental will be)
# added to toast-deps soon.  For example, if I had elemental installed
# to the same prefix I was using for toast, I might run this script with:
#
# $> ./platforms/edison_gnu_mkl.sh --with-elemental=$SCRATCH/software/toastcpp \
#    --prefix=$SCRATCH/software/toastcpp
#

OPTS="$@"

export CC=cc
export CXX=CC
export MPICC=cc
export MPICXX=CC
export CXXFLAGS="-O3"

./configure "${OPTS}" \
--with-mkl="${INTEL_PATH}/linux/mkl/lib/intel64"

