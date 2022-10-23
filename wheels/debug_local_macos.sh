#!/bin/bash

# This script builds against dependencies previously installed by the
# "test_local_macos.sh".  It is useful for repeated build and debug
# cycles against the same dependencies used by the deployed wheels.
#

set -e

# Location of the source tree
pushd $(dirname $0) >/dev/null 2>&1
topdir=$(dirname $(pwd))
popd >/dev/null 2>&1

brew_com=$(which brew)
if [ "x${brew_com}" = "x" ]; then
    echo "Homebrew must be installed and the brew command available"
    exit 1
fi

venv_path=$1
if [ "x${venv_path}" = "x" ]; then
    echo "Usage:  $0 <virtualenv path>"
    echo "  The virtualenv must exist and will be activated."
    exit 1
fi

# Deactivate any current venv
if [ "x$(type -t deactivate)" != "x" ]; then
    deactivate
fi

# Export compiler information
export CC=clang
export CXX=clang++
export CFLAGS="-O3 -fPIC"
export CXXFLAGS="-O3 -fPIC -std=c++11 -stdlib=libc++"

if [ -d ${venv_path} ]; then
    echo "Virtualenv \"${venv_path}\" exists, activating"
    source "${venv_path}/bin/activate"
else
    echo "Virtualenv \"${venv_path}\" does not exist"
    exit 1
fi

# Look for our dependencies in the virtualenv
export LD_LIBRARY_PATH="${venv_path}/lib"
export DYLD_LIBRARY_PATH="${venv_path}/lib"
export CPATH="${venv_path}/include"

# Set up toast build options
export TOAST_BUILD_CMAKE_C_COMPILER="${CC}"
export TOAST_BUILD_CMAKE_CXX_COMPILER="${CXX}"
export TOAST_BUILD_DISABLE_OPENMP=1
export TOAST_BUILD_CMAKE_C_FLAGS="${CFLAGS} -I${venv_path}/include"
export TOAST_BUILD_BLAS_LIBRARIES="${venv_path}/lib/libopenblas.dylib"
export TOAST_BUILD_LAPACK_LIBRARIES="${venv_path}/lib/libopenblas.dylib"
export TOAST_BUILD_CMAKE_CXX_FLAGS="${CXXFLAGS} -I${venv_path}/include"
export TOAST_BUILD_CMAKE_VERBOSE_MAKEFILE=ON
export TOAST_BUILD_AATM_ROOT="${venv_path}"
export TOAST_BUILD_FFTW_ROOT="${venv_path}"
export TOAST_BUILD_SUITESPARSE_INCLUDE_DIR_HINTS="${venv_path}/include"
export TOAST_BUILD_SUITESPARSE_LIBRARY_DIR_HINTS="${venv_path}/lib"
export TOAST_BUILD_CMAKE_LIBRARY_PATH="${venv_path}/lib"

# Now build a wheel
pushd "${topdir}" >/dev/null 2>&1
rm -rf build/temp_wheels/*.whl
python3 setup.py clean
python3 -m pip wheel --wheel-dir=build/temp_wheels --no-deps -vvv .
popd >/dev/null 2>&1

# The file
input_wheel=$(ls ${topdir}/build/temp_wheels/*.whl)
wheel_file=$(basename ${input_wheel})

# Repair it
delocate-listdeps ${input_wheel} \
&& delocate-wheel -w ${topdir} ${input_wheel}

# Install it
python3 -m pip install ${topdir}/${wheel_file}

# Run tests
python3 -c 'import toast.tests; toast.tests.run()'
