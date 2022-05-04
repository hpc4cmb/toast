#!/bin/bash

# Before running this from the toast git checkout directory,
# you shoul pip install cibuildwheel

export CIBW_BUILD="cp39-manylinux_x86_64"
export CIBW_MANYLINUX_X86_64_IMAGE="manylinux2014"
export CIBW_BUILD_VERBOSITY=3
export CIBW_ENVIRONMENT_LINUX="TOAST_BUILD_BLAS_LIBRARIES='-lopenblas -fopenmp -lm -lgfortran' TOAST_BUILD_LAPACK_LIBRARIES='-lopenblas -fopenmp -lm -lgfortran' TOAST_BUILD_CMAKE_VERBOSE_MAKEFILE=ON TOAST_BUILD_TOAST_STATIC_DEPS=ON"
export CIBW_BEFORE_BUILD_LINUX=./wheels/install_deps_linux.sh
export CIBW_BEFORE_TEST="export OMP_NUM_THREADS=2"
export CIBW_TEST_COMMAND=". {project}/wheels/cibw_run_tests.sh"

# Get the current date for logging
now=$(date "+%Y-%m-%d_%H:%M:%S")

# Run it
cibuildwheel --platform linux --output-dir wheelhouse . 2>&1 | tee log_${now}

