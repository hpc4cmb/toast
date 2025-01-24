#!/bin/bash
#
# This script is designed to run within a container managed by cibuildwheel.
# This will use a recent version of OS X.
#
# The purpose of this script is to install TOAST dependency libraries that will be
# bundled with our compiled extension.
#

set -e

prefix=$2

if [ "x${prefix}" = "x" ]; then
    prefix=/usr/local
fi

export PREFIX="${prefix}"

# If we are running on github CI, ensure that permissions
# are set on /usr/local.  See:
# https://github.com/actions/runner-images/issues/9272
sudo chown -R runner:admin /usr/local/

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
echo "Wheel script directory = ${scriptdir}"

# Location of dependency scripts
depsdir=$(dirname ${scriptdir})/deps

# Are we using gcc?  Useful for OpenMP.
# use_gcc=yes
use_gcc=no
gcc_version=14

# Build options.

if [ "x${use_gcc}" = "xyes" ]; then
    export CC=gcc-${gcc_version}
    export CXX=g++-${gcc_version}
    export FC=gfortran-${gcc_version}
    export CFLAGS="-O3 -fPIC"
    export FCFLAGS="-O3 -fPIC"
    export CXXFLAGS="-O3 -fPIC -std=c++11"
    export FCLIBS="-lgfortran"
    export OMPFLAGS="-fopenmp"
else
    # Set the deployment target based on how python was built
    export MACOSX_DEPLOYMENT_TARGET=$(python3 -c "import sysconfig as s; print(s.get_config_vars()['MACOSX_DEPLOYMENT_TARGET'])")
    echo "Using MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}"
    export CC=clang
    export CXX=clang++
    export FC=
    export CFLAGS="-O3 -fPIC"
    export CXXFLAGS="-O3 -fPIC -std=c++11 -stdlib=libc++"
    export FCFLAGS=""
    export FCLIBS=""
    export OMPFLAGS=""
fi

# Install any pre-built dependencies with homebrew

brew install cmake
# Force uninstall flac tools, to avoid conflicts with our
# custom compiled version.
brew uninstall -f --ignore-dependencies flac libogg libsndfile libvorbis opusfile sox
if [ "x${use_gcc}" = "xyes" ]; then
    brew install gcc@${gcc_version}
fi

# Update pip
pip install --upgrade pip

# Install a couple of base packages that are always required
pip install -v wheel

pyver=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")

# Install build requirements.
CC="${CC}" CFLAGS="${CFLAGS}" pip install -v -r "${scriptdir}/build_requirements.txt" numpy

# Build compiled dependencies

export MAKEJ=2
export DEPSDIR="${depsdir}"
export STATIC=no
export SHLIBEXT="dylib"
export CLEANUP=yes

export BLAS_LIBRARIES="-L${PREFIX}/lib -lopenblas ${OMPFLAGS} -lm ${FCLIBS}"
export LAPACK_LIBRARIES="-L${PREFIX}/lib -lopenblas ${OMPFLAGS} -lm ${FCLIBS}"

for pkg in openblas fftw libflac suitesparse libaatm; do
    source "${depsdir}/${pkg}.sh"
done
