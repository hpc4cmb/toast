#!/bin/bash

# Before running this from the toast git checkout directory,
# you shoul pip install cibuildwheel

export CIBW_BUILD="cp38-manylinux_x86_64"
export CIBW_MANYLINUX_X86_64_IMAGE="manylinux2010"
export CIBW_BUILD_VERBOSITY=3
export CIBW_ENVIRONMENT_LINUX="TOAST_BUILD_BLAS_LIBRARIES='-lopenblas -fopenmp -lm -lgfortran' TOAST_BUILD_LAPACK_LIBRARIES='-lopenblas -fopenmp -lm -lgfortran' TOAST_BUILD_CMAKE_VERBOSE_MAKEFILE=ON"
export CIBW_BEFORE_BUILD_LINUX=./wheels/install_deps_linux.sh
export CIBW_BEFORE_TEST="pip3 install numpy && pip3 install mpi4py"
export CIBW_TEST_COMMAND="export OMP_NUM_THREADS=2; python -c 'import toast.tests; toast.tests.run()'"

# Get the current date for logging
now=$(date "+%Y-%m-%d_%H:%M:%S")

# Run it
cibuildwheel --platform linux --archs x86_64 --output-dir wheelhouse . 2>&1 | tee log_${now}

