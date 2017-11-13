#!/bin/bash

# This script assumes that the "toast-deps" module has been loaded.
# To install to my scratch directory, I might run this script with:
#
# $> ./cmake/platforms/edison-gcc_mkl.sh \
#    -DCMAKE_PREFIX_PATH=$SCRATCH/software/toast-gcc
#

ROOT=${PWD}
DIR=${PWD}/build-toast/edison-gcc-mkl/release
OPTS="$@"

export CC=$(which cc)
export CXX=$(which CC)
export MPICC=$(which cc)
export MPICXX=$(which CC)

# ${OPTS} at end because last -D overrides any previous -D with same name
# on command line (e.g. -DUSE_MKL=ON -DUSE_MKL=OFF --> USE_MKL == OFF)
mkdir -p ${DIR} 
cd ${DIR}
cmake -DCMAKE_BUILD_TYPE=Release \
    -DUSE_MKL=ON -DMKL_ROOT=${INTEL_PATH}/linux/mkl \
    -DUSE_TBB=ON -DTBB_ROOT=${INTEL_PATH}/linux/tbb \
    ${OPTS} ${ROOT}

