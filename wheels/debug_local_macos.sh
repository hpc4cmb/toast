#!/bin/bash

# This script tests the local use of setup.py on MacOS.  It assumes
# that you have homebrew installed and have the "brew" command in
# your PATH.
#
# This script takes one argument, which is the path to the virtualenv
# to use for installation.  If that path does not exist, it will be
# created.
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
    echo "  If the path to the virtualenv exists, it will be activated."
    echo "  Otherwise it will be created."
    exit 1
fi

# Deactivate any current venv
if [ "x$(type -t deactivate)" != "x" ]; then
    deactivate
fi

# Path to homebrew
brew_root=$(dirname $(dirname ${brew_com}))
echo "Using homebrew installation in ${brew_root}"

# Export compiler information
export CC=clang
export CXX=clang++
export CFLAGS="-O3 -fPIC"
export CXXFLAGS="-O3 -fPIC -std=c++11 -stdlib=libc++"

# Use homebrew python for development files
brew_py="${brew_root}/opt/python@3/bin/python3"
brew_py_ver=$(eval ${brew_py} --version | awk '{print $2}')
echo "Using python ${brew_py}, version ${brew_py_ver}"

if [ -d ${venv_path} ]; then
    echo "Virtualenv \"${venv_path}\" exists, activating"
    source "${venv_path}/bin/activate"
    venv_py_ver=$(python3 --version | awk '{print $2}')
    if [ "${venv_py_ver}" != "${brew_py_ver}" ]; then
        echo "Virtualenv python version ${venv_py_ver} does not match"
        echo "homebrew python version ${brew_py_ver}.  Remove this"
        echo "virtualenv or use a different path."
        exit 1
    fi
else
    echo "Virtualenv \"${venv_path}\" does not exist, run test_local_macos.sh."
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
