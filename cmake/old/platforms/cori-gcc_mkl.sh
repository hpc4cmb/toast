#!/bin/bash

# This script assumes that the "toast-deps" module has been loaded.
# To install to my scratch directory, I might run this script with:
#
# $> ./cmake/platforms/cori-gcc_mkl.sh \
#    -DCMAKE_INSTALL_PREFIX=$SCRATCH/software/toast-gcc
#

set -o errexit
for i in $(env | grep aux | grep ^PATH | sed 's/:/ /g' | sed 's/PATH=//g')
do 
    if [ ! -z "$(echo $i | grep 'aux/bin')" ]; then 
        : ${AUX_PREFIX:=$(dirname $i)}
        break
    fi
done

if [ -z "${AUX_PREFIX}" ]; then 
    echo "Please set AUX_PREFIX to the TOAST auxilary path"
    exit 1
fi

ROOT=${PWD}
DIR=${PWD}/build-toast/cori-gcc-mkl/release
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
    -DUSE_TBB=OFF -DUSE_ARCH=ON -DUSE_SSE=ON -DUSE_AATM=ON \
    -DUSE_ELEMENTAL=ON -DCMAKE_PREFIX_PATH=${AUX_PREFIX} \
    -DCTEST_SITE=cori-knl \
    -DMACHINE=cori-knl \
    ${OPTS} ${ROOT}
