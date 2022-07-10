#!/bin/bash
#
# This script is designed to run within a container managed by cibuildwheel.
# This will run in a manylinux2014 (CentOS 7) container.
#
# The purpose of this script is to install TOAST dependency libraries that will be
# bundled with our compiled extension.
#

set -e

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
echo "Wheel script directory = ${scriptdir}"

# Build options

CC=gcc
CXX=g++
FC=gfortran

CFLAGS="-O3 -fPIC -pthread"
FCFLAGS="-O3 -fPIC -pthread"
CXXFLAGS="-O3 -fPIC -pthread -std=c++11"

MAKEJ=2

PREFIX=/usr

# Use yum to install OS packages

yum update -y

# Update pip
pip install --upgrade pip

# Install a couple of base packages that are always required
pip install -v cmake wheel

# In order to maximize ABI compatibility with numpy, build with the newest numpy
# version containing the oldest ABI version compatible with the python we are using.
pyver=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
if [ ${pyver} == "3.7" ]; then
    numpy_ver="1.20"
fi
if [ ${pyver} == "3.8" ]; then
    numpy_ver="1.20"
fi
if [ ${pyver} == "3.9" ]; then
    numpy_ver="1.20"
fi
if [ ${pyver} == "3.10" ]; then
    numpy_ver="1.22"
fi

# Install build requirements.
CC="${CC}" CFLAGS="${CFLAGS}" pip install -v "numpy<${numpy_ver}" -r "${scriptdir}/build_requirements.txt"

# Install openblas from the multilib package- the same one numpy uses.

openblas_pkg="openblas-v0.3.20-manylinux2014_x86_64.tar.gz"
openblas_url="https://anaconda.org/multibuild-wheels-staging/openblas-libs/v0.3.20/download/${openblas_pkg}"

if [ ! -e ${openblas_pkg} ]; then
    echo "Fetching OpenBLAS..."
    curl -SL ${openblas_url} -o ${openblas_pkg}
fi

echo "Extracting OpenBLAS"
tar -x -z -v -C "${PREFIX}" --strip-components 2 -f ${openblas_pkg}

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
    ./configure \
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
    --disable-fortran \
    --enable-openmp \
    --enable-shared \
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

ssparse_version=5.12.0
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
    && patch -p1 < "${scriptdir}/suitesparse.patch" \
    && make library JOBS=${MAKEJ} INSTALL="${PREFIX}" \
    CC="${CC}" CXX="${CXX}" \
    CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" AUTOCC=no \
    GPU_CONFIG="" CFOPENMP="${OPENMP_CXXFLAGS}" BLAS="-lopenblas -lm" \
    LAPACK="-lopenblas -lm" \
    && make install INSTALL="${PREFIX}" \
    CC="${CC}" CXX="${CXX}" \
    CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" AUTOCC=no \
    GPU_CONFIG="" CFOPENMP="${OPENMP_CXXFLAGS}" BLAS="-lopenblas -lm" \
    LAPACK="-lopenblas -lm" \
    && popd >/dev/null 2>&1

# MPI for testing.  Although we do not bundle MPI with toast, we need
# to install it in order to run mpi-enabled tests on the produced
# wheel.

mpich_version=3.4
mpich_dir=mpich-${mpich_version}
mpich_pkg=${mpich_dir}.tar.gz

echo "Fetching MPICH..."

if [ ! -e ${mpich_pkg} ]; then
    curl -SL https://www.mpich.org/static/downloads/${mpich_version}/${mpich_pkg} -o ${mpich_pkg}
fi

echo "Building MPICH..."

tar xzf ${mpich_pkg} \
    && cd ${mpich_dir} \
    && CC="${CC}" CXX="${CXX}" FC="${FC}" F77="${FC}" \
    CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" \
    FFLAGS="${FCFLAGS}" FCFLAGS="${FCFLAGS}" \
    MPICH_MPICC_CFLAGS="${CFLAGS}" \
    MPICH_MPICXX_CXXFLAGS="${CXXFLAGS}" \
    MPICH_MPIF77_FFLAGS="${FCFLAGS}" \
    MPICH_MPIFORT_FCFLAGS="${FCFLAGS}" \
    ./configure --disable-fortran \
    --with-device=ch3 \
    --prefix="${PREFIX}" \
    && make -j ${MAKEJ} \
    && make install

# Install mpi4py for running tests

echo "mpicc = $(which mpicc)"
echo "mpicxx = $(which mpicxx)"
echo "mpirun = $(which mpirun)"
python3 -m pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
