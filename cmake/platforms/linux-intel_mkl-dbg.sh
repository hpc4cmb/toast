#!/bin/bash

# Pass extra configure options to this script, including
# things like -DCMAKE_INSTALL_PREFIX=, -DUSE_ELEMENTAL=ON, -DElemental_ROOT=/opt/elemental, etc.
#
# NOTE:
#   - Specifying Elemental_ROOT in the command line or in the enviroment will execute
#     local "FindElemental.cmake", however, Elemental installed will cmake will
#     contain a CMake export file. The export file is maintained by the Elemental
#     developers and is recommended to use over "FindElemental.cmake"
#     - To use export file, specify the directory containing "ElementalConfig.cmake"
#       in the Elemental install tree (typically located at
#         ${CMAKE_INSTALL_PREFIX}/CMake/elemental
#     - E.g.:
#        ./cmake/platforms/linux-intel-mkl-dbg.sh -DElemental_DIR=/opt/elemental/CMake/elemental 

ROOT=${PWD}
DIR=${PWD}/build-toast/linux-intel-mkl/debug
OPTS="$@"

export CC=$(which icc)
export CXX=$(which icpc)
export MPICC=$(which icc)
export MPICXX=$(which icpc)

# ${OPTS} at end because last -D overrides any previous -D with same name
# on command line (e.g. -DUSE_MPI=OFF -DUSE_MPI=OFF --> USE_MPI == OFF)
mkdir -p ${DIR} 
cd ${DIR}
cmake -DCMAKE_BUILD_TYPE=Debug \
    -DUSE_MKL=ON -DMKL_ROOT=${INTEL_PATH}/linux/mkl \
    -DUSE_TBB=ON -DTBB_ROOT=${INTEL_PATH}/linux/tbb \
    -DUSE_MATH=ON -DIMF_ROOT=${INTEL_PATH}/linx/compiler \
    -DTARGET_ARCHITECTURE=auto \
    ${OPTS} ${ROOT}

