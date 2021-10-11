#!/bin/bash
#
# This script is designed to run within a container managed by cibuildwheel.
# This will use a recent version of OS X.
#
# The purpose of this script is to install TOAST dependency libraries that will be
# bundled with our compiled extension.
#

set -e

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
topdir=$(pwd)
popd >/dev/null 2>&1

# Install mpich and mpi4py
brew install mpich
pip install mpi4py

# Get newer cmake with pip
pip install cmake

# Build options

CC=clang
CXX=clang++

CFLAGS="-O3 -fPIC"
CXXFLAGS="-O3 -fPIC -std=c++11 -stdlib=libc++"

MAKEJ=2

PREFIX=/usr/local

# libgmp

gmp_version=6.2.1
gmp_dir=gmp-${gmp_version}
gmp_pkg=${gmp_dir}.tar.xz

echo "Fetching libgmp"

if [ ! -e ${gmp_pkg} ]; then
    curl -SL https://ftp.gnu.org/gnu/gmp/${gmp_pkg} -o ${gmp_pkg}
fi

echo "Building libgmp..."

rm -rf ${gmp_dir}
tar xf ${gmp_pkg} \
    && pushd ${gmp_dir} >/dev/null 2>&1 \
    && CC="${CC}" CFLAGS="${CFLAGS}" \
    && CXX="${CXX}" CXXFLAGS="${CXXFLAGS}" \
    ./configure \
    --enable-static \
    --disable-shared \
    --with-pic \
    --prefix="${PREFIX}" \
    && make -j ${MAKEJ} \
    && make install \
    && popd >/dev/null 2>&1

# libmpfr

mpfr_version=4.1.0
mpfr_dir=mpfr-${mpfr_version}
mpfr_pkg=${mpfr_dir}.tar.xz

echo "Fetching libmpfr"

if [ ! -e ${mpfr_pkg} ]; then
    curl -SL https://www.mpfr.org/mpfr-current/${mpfr_pkg} -o ${mpfr_pkg}
fi

echo "Building libmpfr..."

rm -rf ${mpfr_dir}
tar xf ${mpfr_pkg} \
    && pushd ${mpfr_dir} >/dev/null 2>&1 \
    && CC="${CC}" CFLAGS="${CFLAGS}" \
    ./configure \
    --enable-static \
    --disable-shared \
    --with-pic \
    --with-gmp="${PREFIX}" \
    --prefix="${PREFIX}" \
    && make -j ${MAKEJ} \
    && make install \
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

rm -rf ${fftw_dir}
tar xzf ${fftw_pkg} \
    && pushd ${fftw_dir} >/dev/null 2>&1 \
    && CC="${CC}" CFLAGS="${CFLAGS}" \
    ./configure \
    --enable-threads \
    --enable-static \
    --disable-shared \
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

ssparse_version=5.10.1
ssparse_dir=SuiteSparse-${ssparse_version}
ssparse_pkg=${ssparse_dir}.tar.gz

echo "Fetching SuiteSparse..."

if [ ! -e ${ssparse_pkg} ]; then
    curl -SL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v${ssparse_version}.tar.gz -o ${ssparse_pkg}
fi

echo "Building SuiteSparse..."

rm -rf ${ssparse_dir}
tar xzf ${ssparse_pkg} \
    && pushd ${ssparse_dir} >/dev/null 2>&1 \
    && patch -p1 < "${topdir}/suitesparse.patch" \
    && make library JOBS=${MAKEJ} \
    CC="${CC}" CXX="${CXX}" \
    CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" AUTOCC=no \
    GPU_CONFIG="" BLAS="-framework Accelerate" \
    && make static JOBS=${MAKEJ} \
    CC="${CC}" CXX="${CXX}" \
    CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" AUTOCC=no \
    GPU_CONFIG="" BLAS="-framework Accelerate" \
    && cp -a ./include/* "${PREFIX}/include/" \
    && find . -name "*.a" -exec cp -a '{}' "${PREFIX}/lib/" \; \
    && popd >/dev/null 2>&1

# This line removed from above, since we are linking to static libs:
# && cp -a ./lib/* "${PREFIX}/lib/" \
