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
PREFIX=$2

MAKEJ=2

if [ "x${PREFIX}" = "x" ]; then
    PREFIX=/usr/local
fi

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

# Are we using gcc?  Useful for OpenMP.
# use_gcc=yes
use_gcc=no
gcc_version=12

# Build options.

if [ "x${use_gcc}" = "xyes" ]; then
    CC=gcc-${gcc_version}
    CXX=g++-${gcc_version}
    FC=gfortran-${gcc_version}
    CFLAGS="-O3 -fPIC"
    FCFLAGS="-O3 -fPIC"
    CXXFLAGS="-O3 -fPIC -std=c++11"
    LIBGFORTRAN="-lgfortran"
else
    CC=clang
    CXX=clang++
    FC=
    CFLAGS="-O3 -fPIC"
    CXXFLAGS="-O3 -fPIC -std=c++11 -stdlib=libc++"
    FCFLAGS=""
    if [ "${arch}" = "macosx_arm64" ]; then
        # We are cross compiling
        CFLAGS="${CFLAGS} -arch arm64"
        CXXFLAGS="${CXXFLAGS} -arch arm64"
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

# OpenBLAS- Kept here for reference, but for clang compilation it requires
# using gfortran and then later convincing clang to link to libgfortran...

# openblas_version=0.3.21
# openblas_dir=OpenBLAS-${openblas_version}
# openblas_pkg=${openblas_dir}.tar.gz

# echo "Fetching OpenBLAS"

# if [ ! -e ${openblas_pkg} ]; then
#     curl -SL https://github.com/xianyi/OpenBLAS/archive/v${openblas_version}.tar.gz -o ${openblas_pkg}
# fi

# echo "Building OpenBLAS..."

# omp_opt="0"
# ld_opt="-lm ${LIBGFORTRAN}"
# if [ "x${use_gcc}" = "xyes" ]; then
#     omp_opt="1"
#     ld_opt="-fopenmp -lm ${LIBGFORTRAN}"
# fi

# rm -rf ${openblas_dir}
# tar xzf ${openblas_pkg} \
#     && pushd ${openblas_dir} >/dev/null 2>&1 \
#     && make USE_OPENMP=${omp_opt} NO_SHARED=1 \
#     MAKE_NB_JOBS=${MAKEJ} \
#     CC="${CC}" FC="${FC}" DYNAMIC_ARCH=1 TARGET=GENERIC \
#     COMMON_OPT="${CFLAGS}" FCOMMON_OPT="${FCFLAGS}" \
#     LDFLAGS="${ld_opt}" libs netlib \
#     && make NO_SHARED=1 DYNAMIC_ARCH=1 TARGET=GENERIC PREFIX="${PREFIX}" install \
#     && popd >/dev/null 2>&1

# Install openblas from the multilib package- the same one numpy uses.

if [ "${arch}" = "macosx_arm64" ]; then
    openblas_pkg="openblas-v0.3.21-macosx_11_0_arm64-gf_5272328.tar.gz"
else
    openblas_pkg="openblas-v0.3.21-macosx_10_9_x86_64-gf_1becaaa.tar.gz"
fi
openblas_url="https://anaconda.org/multibuild-wheels-staging/openblas-libs/v0.3.21/download/${openblas_pkg}"

if [ ! -e ${openblas_pkg} ]; then
    echo "Fetching OpenBLAS..."
    curl -SL ${openblas_url} -o ${openblas_pkg}
fi

echo "Extracting OpenBLAS"
tar -x -z -v -C "${PREFIX}" --strip-components 2 -f ${openblas_pkg}

# Install the gfortran (and libgcc) that was used for openblas compilation

curl -L https://github.com/MacPython/gfortran-install/raw/master/archives/gfortran-4.9.0-Mavericks.dmg -o gfortran.dmg
GFORTRAN_SHA256=$(shasum -a 256 gfortran.dmg)
KNOWN_SHA256="d2d5ca5ba8332d63bbe23a07201c4a0a5d7e09ee56f0298a96775f928c3c4b30  gfortran.dmg"
if [ "$GFORTRAN_SHA256" != "$KNOWN_SHA256" ]; then
    echo sha256 mismatch
    exit 1
fi

hdiutil attach -mountpoint /Volumes/gfortran gfortran.dmg
sudo installer -pkg /Volumes/gfortran/gfortran.pkg -target /
otool -L /usr/local/gfortran/lib/libgfortran.3.dylib

# Install FFTW

fftw_version=3.3.10
fftw_dir=fftw-${fftw_version}
fftw_pkg=${fftw_dir}.tar.gz

echo "Fetching FFTW..."

if [ ! -e ${fftw_pkg} ]; then
    curl -SL http://www.fftw.org/${fftw_pkg} -o ${fftw_pkg}
fi

echo "Building FFTW..."

thread_opt="--enable-threads"
if [ "x${use_gcc}" = "xyes" ]; then
    thread_opt="--enable-openmp"
fi

rm -rf ${fftw_dir}
tar xzf ${fftw_pkg} \
    && pushd ${fftw_dir} >/dev/null 2>&1 \
    && CC="${CC}" CFLAGS="${CFLAGS}" \
    ./configure ${cross} ${thread_opt} \
    --disable-fortran \
    --enable-shared \
    --disable-static \
    --prefix="${PREFIX}" \
    && make -j ${MAKEJ} \
    && make install \
    && popd >/dev/null 2>&1

# Install libaatm

aatm_version=1.0.9
aatm_dir=libaatm-${aatm_version}
aatm_pkg=${aatm_dir}.tar.gz

echo "Fetching libaatm..."

if [ ! -e ${aatm_pkg} ]; then
    curl -SL "https://github.com/hpc4cmb/libaatm/archive/${aatm_version}.tar.gz" -o "${aatm_pkg}"
fi

echo "Building libaatm..."

rm -rf ${aatm_dir}
tar xzf ${aatm_pkg} \
    && pushd ${aatm_dir} >/dev/null 2>&1 \
    && mkdir -p build \
    && pushd build >/dev/null 2>&1 \
    && cmake \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="${CFLAGS}" \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    .. \
    && make -j ${MAKEJ} install \
    && popd >/dev/null 2>&1 \
    && popd >/dev/null 2>&1

# Install SuiteSparse

ssparse_version=6.0.3
ssparse_dir=SuiteSparse-${ssparse_version}
ssparse_pkg=${ssparse_dir}.tar.gz

echo "Fetching SuiteSparse..."

if [ ! -e ${ssparse_pkg} ]; then
    curl -SL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v${ssparse_version}.tar.gz -o ${ssparse_pkg}
fi

echo "Building SuiteSparse..."

if [ "x${use_gcc}" = "xyes" ]; then
    blas_extra="-lgomp ${LIBGFORTRAN}"
else
    blas_extra=""
fi

blas_opt="-L${PREFIX}/lib -lopenblas -lm ${blas_extra}"

rm -rf ${ssparse_dir}
tar xzf ${ssparse_pkg} \
    && pushd ${ssparse_dir} >/dev/null 2>&1 \
    && patch -p1 < "${scriptdir}/suitesparse.patch" \
    && CC="${CC}" CX="${CXX}" JOBS=${MAKEJ} \
    CMAKE_OPTIONS=" \
    -DCMAKE_C_COMPILER=\"${CC}\" \
    -DCMAKE_CXX_COMPILER=\"${CXX}\" \
    -DCMAKE_Fortran_COMPILER=\"${FC}\" \
    -DCMAKE_C_FLAGS=\"${CFLAGS}\" \
    -DCMAKE_CXX_FLAGS=\"${CXXFLAGS}\" \
    -DCMAKE_Fortran_FLAGS=\"${FCFLAGS}\" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_INSTALL_PATH=\"${PREFIX}\" \
    -DNSTATIC:BOOL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLA_VENDOR=OpenBLAS \
    -DBLA_STATIC:BOOL=OFF \
    -DBLAS_LIBRARIES=\"${blas_opt}\" \
    -DLAPACK_LIBRARIES=\"${blas_opt}\" \
    " make local \
    && make install \
    && cp ./lib/*.a ./lib/*.dylib "${PREFIX}/lib/" \
    && cp ./include/* "${PREFIX}/include/" \
    && popd >/dev/null 2>&1
