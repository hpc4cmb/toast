#!/bin/bash
#
# This script is designed to run within a container managed by cibuildwheel.
# This will run in a manylinux2014 (CentOS 7) container.
#
# The purpose of this script is to install TOAST dependency libraries that will be
# bundled with our compiled extension.
#

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
topdir=$(pwd)
popd >/dev/null 2>&1

# Install xz

yum -y install xz

# Get newer cmake with pip
pip install cmake

# Build options

CC=gcc
CXX=g++
FC=gfortran

CFLAGS="-O3 -fPIC -pthread"
FCFLAGS="-O3 -fPIC -pthread"
CXXFLAGS="-O3 -fPIC -pthread -std=c++11"

MAKEJ=2

PREFIX=/usr

# Install MPICH

mpich_version=3.2
mpich_dir=mpich-${mpich_version}
mpich_pkg=${mpich_dir}.tar.gz

echo "Fetching MPICH..."

if [ ! -e ${mpich_pkg} ]; then
    curl -SL http://www.mpich.org/static/downloads/${mpich_version}/${mpich_pkg} -o ${mpich_pkg}
fi

echo "Building MPICH..."

rm -rf ${mpich_dir}
tar xzf ${mpich_pkg} \
    && pushd ${mpich_dir} >/dev/null 2>&1 \
    && CC="${CC}" CXX="${CXX}" \
    CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" \
    ./configure --disable-fortran --prefix="${PREFIX}" \
    && make -j ${MAKEJ} \
    && make install \
    && popd >/dev/null 2>&1

# Install mpi4py

pip install mpi4py

# libgmp

gmp_version=6.2.0
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
    --with-gmp="${PREFIX}" \
    --prefix="${PREFIX}" \
    && make -j ${MAKEJ} \
    && make install \
    && popd >/dev/null 2>&1

# Install Openblas

openblas_version=0.3.10
openblas_dir=OpenBLAS-${openblas_version}
openblas_pkg=${openblas_dir}.tar.gz

echo "Fetching OpenBLAS..."

if [ ! -e ${openblas_pkg} ]; then
    curl -SL https://github.com/xianyi/OpenBLAS/archive/v${openblas_version}.tar.gz -o ${openblas_pkg}
fi

echo "Building OpenBLAS..."

rm -rf ${openblas_dir}
tar xzf ${openblas_pkg} \
    && pushd ${openblas_dir} >/dev/null 2>&1 \
    && make USE_OPENMP=1 NO_SHARED=1 \
    MAKE_NB_JOBS=${MAKEJ} \
    CC="${CC}" FC="${FC}" DYNAMIC_ARCH=1 \
    COMMON_OPT="${CFLAGS}" FCOMMON_OPT="${FCFLAGS}" \
    LDFLAGS="-fopenmp -lm" \
    && make NO_SHARED=1 PREFIX="${PREFIX}" install \
    && popd >/dev/null 2>&1

# Install FFTW

fftw_version=3.3.8
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

ssparse_version=5.8.1
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
    GPU_CONFIG="" CFOPENMP="${OPENMP_CXXFLAGS}" BLAS="-lopenblas -fopenmp -lm" \
    && make static JOBS=${MAKEJ} \
    CC="${CC}" CXX="${CXX}" \
    CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" AUTOCC=no \
    GPU_CONFIG="" CFOPENMP="${OPENMP_CXXFLAGS}" BLAS="-lopenblas -fopenmp -lm" \
    && cp -a ./include/* "${PREFIX}/include/" \
    && find . -name "*.a" -exec cp -a '{}' "${PREFIX}/lib/" \; \
    && popd >/dev/null 2>&1

# This line removed from above, since we are linking to static libs:
# && cp -a ./lib/* "${PREFIX}/lib/" \
