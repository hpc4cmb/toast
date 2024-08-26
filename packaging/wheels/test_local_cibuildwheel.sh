#!/bin/bash

# Before running this from the toast git checkout directory,
# you should pip install cibuildwheel

export CIBW_BUILD="cp310-manylinux_x86_64"
export CIBW_MANYLINUX_X86_64_IMAGE="manylinux2014"
export CIBW_BUILD_VERBOSITY=3
export CIBW_ENVIRONMENT_LINUX="TOAST_BUILD_BLAS_LIBRARIES='-L/usr/local/lib -lopenblas -fopenmp -lm -lgfortran' TOAST_BUILD_LAPACK_LIBRARIES='-L/usr/local/lib -lopenblas -fopenmp -lm -lgfortran' TOAST_BUILD_CMAKE_VERBOSE_MAKEFILE=ON TOAST_BUILD_TOAST_STATIC_DEPS=ON TOAST_BUILD_FFTW_ROOT=/usr/local TOAST_BUILD_AATM_ROOT=/usr/local TOAST_BUILD_SUITESPARSE_INCLUDE_DIR_HINTS=/usr/local/include TOAST_BUILD_SUITESPARSE_LIBRARY_DIR_HINTS=/usr/local/lib"
export CIBW_BEFORE_BUILD_LINUX=./packaging/wheels/install_deps_linux.sh
export CIBW_BEFORE_TEST="export OMP_NUM_THREADS=2"
export CIBW_TEST_COMMAND=". {project}/packaging/wheels/cibw_run_tests.sh"

# Get the current date for logging
now=$(date "+%Y-%m-%d_%H:%M:%S")

# Run it
cibuildwheel --platform linux --output-dir wheelhouse . 2>&1 | tee log_${now}

