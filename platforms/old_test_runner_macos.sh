#!/bin/bash

# This script is for a complete build and install to /usr/local inside a
# docker container.  It is used when running CI unit tests.

set -e

# Get the absolute path to the source tree
pushd $(dirname $(dirname $0)) >/dev/null 2>&1
toastdir=$(pwd -P)
popd >/dev/null 2>&1

# Install minimal dependencies
eval ${toastdir}/wheels/install_deps_osx.sh macosx_x86_64 yes

# Install toast

pushd ${toastdir} >/dev/null 2>&1

export TOAST_BUILD_CMAKE_C_COMPILER=clang
export TOAST_BUILD_CMAKE_CXX_COMPILER=clang++
export TOAST_BUILD_BLA_VENDOR='Apple'
export TOAST_BUILD_DISABLE_OPENMP=1
export TOAST_BUILD_CMAKE_C_FLAGS="-O2 -g -fPIC"
export TOAST_BUILD_CMAKE_CXX_FLAGS="-O2 -g -fPIC -std=c++11 -stdlib=libc++"
export TOAST_BUILD_CMAKE_VERBOSE_MAKEFILE=ON
python3 -m pip -vvv install .

popd >/dev/null 2>&1
