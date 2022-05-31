#!/bin/bash

# This script is for a complete build and install to /usr/local inside a
# docker container.  It is used when running CI unit tests.

set -e

# Temporary workaround- update pshmem pip package until upstream container
# has new release.
# python3 -m pip install --upgrade pshmem

# Get the absolute path to the source tree
pushd $(dirname $(dirname $0)) >/dev/null 2>&1
toastdir=$(pwd -P)
popd >/dev/null 2>&1

mkdir build
pushd build >/dev/null 2>&1

cmake \
    -DCMAKE_C_COMPILER="gcc" \
    -DCMAKE_CXX_COMPILER="g++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    "${toastdir}"

make -j 2 install

popd >/dev/null 2>&1
