#!/bin/bash

# For example:
# $> mkdir -p build
# $> cd build
# $> ../platforms/osx_homebrew.sh -DCMAKE_INSTALL_PREFIX=blah
# $> make -j 4
# $> make install
#

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="clang" \
    -DCMAKE_CXX_COMPILER="clang++" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    ${opts} \
    ..
