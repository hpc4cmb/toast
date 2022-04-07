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
export FC=
export CFLAGS="-O3 -fPIC"
export FCFLAGS="-O3 -fPIC"
export CXXFLAGS="-O3 -fPIC -std=c++11 -stdlib=libc++"

# Install most dependencies with homebrew, including python-3.9
eval ${brew_com} install mpich
eval ${brew_com} install python3

# Add this homebrew python to our path
export PATH="${brew_root}/opt/python@3.9/bin:$PATH"
brew_py_ver=$(python3 --version | awk '{print $2}')
echo "Using python $(which python3), version ${brew_py_ver}"

if [ -d ${venv_path} ]; then
    echo "Virtualenv \"${venv_path}\" already exists, activating"
    source "${venv_path}/bin/activate"
    venv_py_ver=$(python3 --version | awk '{print $2}')
    if [ "${venv_py_ver}" != "${brew_py_ver}" ]; then
        echo "Virtualenv python version ${venv_py_ver} does not match"
        echo "homebrew python version ${brew_py_ver}.  Remove this"
        echo "virtualenv or use a different path."
        exit 1
    fi
else
    echo "Creating virtualenv \"${venv_path}\""
    python3 -m venv "${venv_path}"
    source "${venv_path}/bin/activate"
fi

# Install our dependencies
eval "${topdir}/wheels/install_deps_osx.sh"

# Look for our dependencies in /usr/local
export LD_LIBRARY_PATH="/usr/local/lib"
export DYLD_LIBRARY_PATH="/usr/local/lib"
export CPATH="/usr/local/include"
export TOAST_BUILD_CMAKE_LIBRARY_PATH="/usr/local/lib"

# Now build a wheel
python3 setup.py clean
python3 -m pip wheel ${topdir} --wheel-dir=build/temp_wheels --no-deps -vvv

# The file
input_wheel=$(ls ${topdir}/build/temp_wheels/*.whl)
wheel_file=$(basename ${input_wheel})

# Repair it
delocate-listdeps ${input_wheel} \
&& delocate-wheel -w ${topdir} ${input_wheel}

# Install it
python3 -m pip install ${topdir}/${wheel_file}

# Run tests
export OMP_NUM_THREADS=2
python3 -c 'import toast.tests; toast.tests.run()'
