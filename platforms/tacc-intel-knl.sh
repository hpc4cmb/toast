#!/bin/bash

# This script assumes that the "toast-deps" module has been loaded.

OPTS="$@"

export MPICC=mpiicc
export MPICXX=mpiicpc
export CFLAGS="-O3 -g -fPIC -xMIC-AVX512 -pthread"
export CXXFLAGS="-O3 -g -fPIC -xMIC-AVX512 -pthread"
export OPENMP_CFLAGS="-qopenmp"
export OPENMP_CXXFLAGS="-qopenmp"

./configure ${OPTS} \
    --build x86_64-pc-linux-gnu --host x86_64-unknown-linux-gnu \
    --with-math="-limf" \
    --with-mkl="${TACC_INTEL_DIR}/mkl/lib/intel64"

