#!/bin/bash

# This script is for a complete build and install to /usr/local inside a
# docker container.  It is used when running CI unit tests.

set -e

# Get the absolute path to the source tree
pushd $(dirname $(dirname $0)) > /dev/null
toastdir=$(pwd -P)
popd > /dev/null

mkdir build
pushd build

cmake \
    -DCMAKE_C_COMPILER="gcc" \
    -DCMAKE_CXX_COMPILER="g++" \
    -DMPI_C_COMPILER="mpicc" \
    -DMPI_CXX_COMPILER="mpicxx" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread -std=c++11" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    "${toastdir}"

make -j 2 install

popd > /dev/null
