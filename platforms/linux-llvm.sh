#!/bin/bash

# This assumes you have created a virtualenv and installed toast
# dependencies (e.g. with wheels/install_deps_linux.sh) into that virtual
# env.  This assumes installation of llvm-17 from LLVM apt sources.

venv_path=$(dirname $(dirname $(which python3)))

export LD_LIBRARY_PATH="${venv_path}/lib"

cmake \
    -DCMAKE_C_COMPILER="clang-17" \
    -DCMAKE_CXX_COMPILER="clang++-17" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread -I${venv_path}/include" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread -std=c++11 -stdlib=libc++ -I${venv_path}/include" \
    -DUSE_OPENMP_TARGET=TRUE \
    -DOPENMP_TARGET_FLAGS="-fopenmp -fopenmp-targets=nvptx64 -fopenmp-target-debug=3 --offload-arch=sm_86 -Wl,-lomp,-lomptarget,-lomptarget.rtl.cuda" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCMAKE_LIBRARY_PATH="${venv_path}/lib" \
    -DBLAS_LIBRARIES="-L${venv_path}/lib -lopenblas -fopenmp -lm -lgfortran" \
    -DLAPACK_LIBRARIES="-L${venv_path}/lib -lopenblas -fopenmp -lm -lgfortran" \
    -DAATM_ROOT="${venv_path}" \
    -DFFTW_ROOT="${venv_path}" \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="${venv_path}/include" \
    -DSUITESPARSE_LIBRARY_DIR_HINTS="${venv_path}/lib" \
    -DTOAST_STATIC_DEPS=ON \
    -DCMAKE_INSTALL_PREFIX="${venv_path}" \
    ..

