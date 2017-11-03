#!/bin/bash

# This script assumes that the "toast-deps" module has been loaded.
# To install to my scratch directory, I might run this script with:
#
# $> ./cmake/platforms/tacc-knl_intel_mkl.sh \
#    -DCMAKE_PREFIX_PATH=$SCRATCH/software/toast-knl
#

ROOT=${PWD}
DIR=${PWD}/build-toast/tacc-knl-intel-mkl/release
OPTS="$@"

export CC=$(which cc)
export CXX=$(which CC)
export MPICC=$(which cc)
export MPICXX=$(which CC)

# ${OPTS} at end because last -D overrides any previous -D with same name
# on command line (e.g. -DUSE_MPI=OFF -DUSE_MPI=OFF --> USE_MPI == OFF)
mkdir -p ${DIR} 
cd ${DIR}
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON \
    -DUSE_MKL=ON -DMKL_ROOT=${TACC_INTEL_DIR}/linux/mkl \
    -DUSE_TBB=ON -DTBB_ROOT=${TACC_INTEL_DIR}/linux/tbb \
    -DUSE_MATH=ON -DIMF_ROOT=${INTEL_PATH}/linx/compiler \
    -DTARGET_ARCHITECTURE=knl \
    ${OPTS} ${ROOT}


