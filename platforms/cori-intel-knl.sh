#!/bin/bash

# This script assumes that the "toast-deps" module has been loaded.
# To install to my scratch directory, I might run this script with:
#
# $> ./platforms/cori-knl_intel_mkl.sh \
#    --prefix=$SCRATCH/software/toast-knl
#

OPTS="$@"

export CC=cc
export CXX=CC
export MPICC=cc
export MPICXX=CC
export CFLAGS="-O3 -g -fPIC -xMIC-AVX512 -pthread"
export CXXFLAGS="-O3 -g -fPIC -xMIC-AVX512 -pthread"
export OPENMP_CFLAGS="-qopenmp"
export OPENMP_CXXFLAGS="-qopenmp"

./configure ${OPTS} \
    --build x86_64-pc-linux-gnu --host x86_64-unknown-linux-gnu \
    --with-math="-limf" \
    --with-mkl="${INTEL_PATH}/linux/mkl/lib/intel64"

