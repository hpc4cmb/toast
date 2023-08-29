#!/bin/bash
#
# This script is designed to run within a container managed by cibuildwheel.
# This will run in a manylinux2014 (CentOS 7) container.
#
# The purpose of this script is to install TOAST dependency libraries that will be
# bundled with our compiled extension.
#

set -e

toolchain=$1
PREFIX=$2
static=$3

# For testing locally
# MAKEJ=8
MAKEJ=2

if [ "x${toolchain}" = "x" ]; then
    toolchain="gcc"
fi

if [ "x${PREFIX}" = "x" ]; then
    PREFIX=/usr/local
fi

if [ "x${static}" = "x" ]; then
    static="yes"
fi

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
echo "Wheel script directory = ${scriptdir}"

# Location of dependency scripts
depsdir="${scriptdir}/deps"

# Build options

if [ "x${toolchain}" = "xgcc" ]; then
    CC=gcc
    CXX=g++
    FC=gfortran

    CFLAGS="-O3 -fPIC -pthread"
    FCFLAGS="-O3 -fPIC -pthread"
    CXXFLAGS="-O3 -fPIC -pthread -std=c++11"
    FCLIBS="-lgfortran"
else
    if [ "x${toolchain}" = "xllvm" ]; then
        CC=clang-17
        CXX=clang++-17
        FC=gfortran

        CFLAGS="-O3 -fPIC -pthread"
        FCFLAGS="-O3 -fPIC -pthread"
        CXXFLAGS="-O3 -fPIC -pthread -std=c++11 -stdlib=libc++"
        FCLIBS="-L/usr/lib/llvm-17/lib /usr/lib/x86_64-linux-gnu/libgfortran.so.5"
    else
        echo "Unsupported toolchain \"${toolchain}\""
        exit 1
    fi
fi

# Update pip
pip install --upgrade pip

# Install a couple of base packages that are always required
pip install -v cmake wheel

# In order to maximize ABI compatibility with numpy, build with the newest numpy
# version containing the oldest ABI version compatible with the python we are using.
pyver=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
if [ ${pyver} == "3.8" ]; then
    numpy_ver="1.20"
fi
if [ ${pyver} == "3.9" ]; then
    numpy_ver="1.20"
fi
if [ ${pyver} == "3.10" ]; then
    numpy_ver="1.22"
fi
if [ ${pyver} == "3.11" ]; then
    numpy_ver="1.24"
fi

# Install build requirements.
CC="${CC}" CFLAGS="${CFLAGS}" pip install -v "numpy<${numpy_ver}" -r "${scriptdir}/build_requirements.txt"

# Install Openblas

openblas_version=0.3.23
openblas_dir=OpenBLAS-${openblas_version}
openblas_pkg=${openblas_dir}.tar.gz

if [ ! -e ${openblas_pkg} ]; then
    echo "Fetching OpenBLAS..."
    curl -SL https://github.com/xianyi/OpenBLAS/archive/v${openblas_version}.tar.gz -o ${openblas_pkg}
fi

echo "Building OpenBLAS..."

shr="NO_STATIC=1"
if [ "${static}" = "yes" ]; then
    shr="NO_SHARED=1"
fi

rm -rf ${openblas_dir}
tar xzf ${openblas_pkg} \
    && pushd ${openblas_dir} >/dev/null 2>&1 \
    && make USE_OPENMP=1 ${shr} \
    MAKE_NB_JOBS=${MAKEJ} \
    CC="${CC}" FC="${FC}" DYNAMIC_ARCH=1 TARGET=GENERIC \
    COMMON_OPT="${CFLAGS}" FCOMMON_OPT="${FCFLAGS}" \
    EXTRALIB="-fopenmp -lm ${FCLIBS}" all \
    && make ${shr} DYNAMIC_ARCH=1 TARGET=GENERIC PREFIX="${PREFIX}" install \
    && popd >/dev/null 2>&1

# Install FFTW

fftw_version=3.3.10
fftw_dir=fftw-${fftw_version}
fftw_pkg=${fftw_dir}.tar.gz

echo "Fetching FFTW..."

if [ ! -e ${fftw_pkg} ]; then
    curl -SL http://www.fftw.org/${fftw_pkg} -o ${fftw_pkg}
fi

echo "Building FFTW..."

shr="--enable-shared --disable-static"
if [ "${static}" = "yes" ]; then
    shr="--enable-static --disable-shared"
fi

rm -rf ${fftw_dir}
tar xzf ${fftw_pkg} \
    && pushd ${fftw_dir} >/dev/null 2>&1 \
    && CC="${CC}" CFLAGS="${CFLAGS}" \
    ./configure \
    --disable-fortran \
    --enable-openmp ${shr} \
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

# Install libFLAC

flac_version=1.4.3
flac_dir=flac-${flac_version}
flac_pkg=${flac_dir}.tar.gz

echo "Fetching libFLAC..."

if [ ! -e ${flac_pkg} ]; then
    curl -SL "https://github.com/xiph/flac/archive/refs/tags/${flac_version}.tar.gz" -o "${flac_pkg}"
fi

echo "Building libFLAC..."

rm -rf ${flac_dir}
tar xzf ${flac_pkg} \
    && pushd ${flac_dir} >/dev/null 2>&1 \
    && mkdir -p build \
    && pushd build >/dev/null 2>&1 \
    && cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="${CFLAGS}" \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DWITH_OGG=OFF \
    -DINSTALL_MANPAGES=OFF \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    .. \
    && make -j ${MAKEJ} install \
    && popd >/dev/null 2>&1 \
    && popd >/dev/null 2>&1

# Install SuiteSparse - only the pieces we need.

ssparse_version=7.1.0
ssparse_dir=SuiteSparse-${ssparse_version}
ssparse_pkg=${ssparse_dir}.tar.gz

echo "Fetching SuiteSparse..."

if [ ! -e ${ssparse_pkg} ]; then
    curl -SL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v${ssparse_version}.tar.gz -o ${ssparse_pkg}
fi

echo "Building SuiteSparse..."

shr="-DNSTATIC:BOOL=ON -DBLA_STATIC:BOOL=OFF"
if [ "${static}" = "yes" ]; then
    shr="-DNSTATIC:BOOL=OFF -DBLA_STATIC:BOOL=ON"
fi

cmake_opts=" \
    -DCMAKE_C_COMPILER=\"${CC}\" \
    -DCMAKE_CXX_COMPILER=\"${CXX}\" \
    -DCMAKE_Fortran_COMPILER=\"${FC}\" \
    -DCMAKE_C_FLAGS=\"${CFLAGS}\" \
    -DCMAKE_CXX_FLAGS=\"${CXXFLAGS}\" \
    -DCMAKE_Fortran_FLAGS=\"${FCFLAGS}\" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_INSTALL_PATH=\"${PREFIX}\" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLA_VENDOR=OpenBLAS ${shr} \
    -DBLAS_LIBRARIES=\"-L${PREFIX}/lib -lopenblas -lm ${FCLIBS}\" \
    -DLAPACK_LIBRARIES=\"-L${PREFIX}/lib -lopenblas -lm ${FCLIBS}\" \
    "

rm -rf ${ssparse_dir}
tar xzf ${ssparse_pkg} \
    && pushd ${ssparse_dir} >/dev/null 2>&1 \
    && patch -p1 < "${scriptdir}/suitesparse.patch" \
    && for pkg in SuiteSparse_config AMD CAMD CCOLAMD COLAMD CHOLMOD; do \
    pushd ${pkg} >/dev/null 2>&1; \
    CC="${CC}" CX="${CXX}" JOBS=${MAKEJ} \
    CMAKE_OPTIONS=${cmake_opts} \
    make local; \
    make install; \
    popd >/dev/null 2>&1; \
    done \
    && if [ "${static}" = "yes" ]; then cp ./lib/*.a "${PREFIX}/lib/"; \
    else cp ./lib/*.so "${PREFIX}/lib/"; fi \
    && cp ./include/* "${PREFIX}/include/" \
    && popd >/dev/null 2>&1
