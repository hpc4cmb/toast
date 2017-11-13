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
#        ./cmake/platforms/linux-gcc.sh -DElemental_DIR=/opt/elemental/CMake/elemental 

ROOT=${PWD}
DIR=${PWD}/build-toast/linux-gcc/release
OPTS="$@"

export CC=$(which gcc)
export CXX=$(which g++)
export MPICC=$(which gcc)
export MPICXX=$(which g++)

# ${OPTS} at end because last -D overrides any previous -D with same name
# on command line (e.g. -DUSE_MKL=ON -DUSE_MKL=OFF --> USE_MKL == OFF)
mkdir -p ${DIR} 
cd ${DIR}
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=OFF -DUSE_TBB=OFF ${OPTS} ${ROOT}

