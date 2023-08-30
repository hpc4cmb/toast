#!/bin/bash
#
# This script is designed to run within a container managed by cibuildwheel.
# This will use a recent version of OS X.
#
# The purpose of this script is to install TOAST dependency libraries that will be
# bundled with our compiled extension.
#

set -e

arch=$1
prefix=$2

if [ "x${prefix}" = "x" ]; then
    prefix=/usr/local
fi

export PREFIX="${prefix}"

# Cross compile option needed for autoconf builds.
cross=""
if [ "${arch}" = "macosx_arm64" ]; then
    cross="--host=arm64-apple-darwin20.6.0"
fi

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
gcc_version=12

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
    export CC=clang
    export CXX=clang++
    export FC=
    export CFLAGS="-O3 -fPIC"
    export CXXFLAGS="-O3 -fPIC -std=c++11 -stdlib=libc++"
    export FCFLAGS=""
    export FCLIBS=""
    export OMPFLAGS=""
    if [ "${arch}" = "macosx_arm64" ]; then
        # We are cross compiling
        export CFLAGS="${CFLAGS} -arch arm64"
        export CXXFLAGS="${CXXFLAGS} -arch arm64"
    fi
fi

# Install any pre-built dependencies with homebrew
brew install cmake
if [ "x${use_gcc}" = "xyes" ]; then
    brew install gcc@${gcc_version}
fi

# Update pip
pip install --upgrade pip

# Install a couple of base packages that are always required
pip install -v wheel

# In order to maximize ABI compatibility with numpy, build with the newest numpy
# version containing the oldest ABI version compatible with the python we are using.
pyver=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
if [ ${pyver} == "3.8" ]; then
    numpy_ver="1.20"
fi
if [ ${pyver} == "3.9" ]; then
    numpy_ver="1.24"
fi
if [ ${pyver} == "3.10" ]; then
    numpy_ver="1.24"
fi
if [ ${pyver} == "3.11" ]; then
    numpy_ver="1.24"
fi

# Install build requirements.
CC="${CC}" CFLAGS="${CFLAGS}" pip install -v "numpy<${numpy_ver}" -r "${scriptdir}/build_requirements.txt"

# Install openblas from the multilib package- the same one numpy uses.

if [ "${arch}" = "macosx_arm64" ]; then
    openblas_pkg="openblas-v0.3.23-246-g3d31191b-macosx_11_0_arm64-gf_5272328.tar.gz"
else
    openblas_pkg="openblas-v0.3.23-246-g3d31191b-macosx_10_9_x86_64-gf_c469a42.tar.gz"
fi
openblas_url="https://anaconda.org/multibuild-wheels-staging/openblas-libs/v0.3.23-246-g3d31191b/download/${openblas_pkg}"

if [ ! -e ${openblas_pkg} ]; then
    echo "Fetching OpenBLAS..."
    curl -SL ${openblas_url} -o ${openblas_pkg}
fi

echo "Extracting OpenBLAS"
tar -x -z -v -C "${PREFIX}" --strip-components 2 -f ${openblas_pkg}

# Install the gfortran (and libgcc) that was used for openblas compilation

if [ "${arch}" = "macosx_arm64" ]; then
    gfortran_arch=arm64
    known_hash="0d5c118e5966d0fb9e7ddb49321f63cac1397ce8"
else
    gfortran_arch=x86_64
    known_hash="c469a420d2d003112749dcdcbe3c684eef42127e"
fi
gfortran_pkg="gfortran-darwin-${gfortran_arch}-native.tar.gz"
gfortran_url="https://github.com/isuruf/gcc/releases/download/gcc-11.3.0-2/${gfortran_pkg}"

curl -L -O ${gfortran_url}
gfortran_hash=$(shasum ${gfortran_pkg} | awk '{print $1}')

if [ "${gfortran_hash}" != "${known_hash}" ]; then
    echo "gfortran sha256 mismatch: ${gfortran_hash} != ${known_hash}"
    exit 1
fi

sudo mkdir -p /opt
sudo mv ${gfortran_pkg} /opt/
pushd /opt
sudo tar -xvf ${gfortran_pkg}
sudo rm ${gfortran_pkg}
popd

for f in libgfortran.dylib libgfortran.5.dylib libgcc_s.1.dylib libgcc_s.1.1.dylib libquadmath.dylib libquadmath.0.dylib; do
    sudo ln -sf "/opt/gfortran-darwin-${gfortran_arch}-native/lib/$f" "/usr/local/lib/$f"
done
sudo ln -sf "/opt/gfortran-darwin-${gfortran_arch}-native/bin/gfortran" "/usr/local/bin/gfortran"

# Build compiled dependencies

export MAKEJ=2
export DEPSDIR="${depsdir}"
export STATIC=no
export SHLIBEXT="dylib"
export CLEANUP=yes

for pkg in fftw libflac suitesparse libaatm; do
    source "${depsdir}/${pkg}.sh"
done
