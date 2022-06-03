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

# If this arg is set to any value, do not install MPI or atmosphere
# tools.  This is used when running quick unit tests on every PR.
quick=$2

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

# Build options.  If we use clang, then use accelerate framework.  Otherwise
# build and use OpenBLAS.

use_gcc=no

if [ "x${use_gcc}" = "xyes" ]; then
    CC=gcc-11
    CXX=g++-11
    FC=gfortran-11
    CFLAGS="-O3 -fPIC"
    FCFLAGS="-O3 -fPIC"
    CXXFLAGS="-O3 -fPIC -std=c++11"
else
    CC=clang
    CXX=clang++
    FC=
    CFLAGS="-O3 -fPIC"
    FCFLAGS=""
    CXXFLAGS="-O3 -fPIC -std=c++11 -stdlib=libc++"
    if [ "${arch}" = "macosx_arm64" ]; then
        # We are cross compiling
        CFLAGS="${CFLAGS} -arch arm64"
        CXXFLAGS="${CXXFLAGS} -arch arm64"
    fi
fi

MAKEJ=2

PREFIX=/usr/local

# Install OS packages with homebrew

# Optionally install gcc
if [ "x${use_gcc}" = "xyes" ]; then
    brew install gcc
fi

# NOTE:  this is disabled, since the homebrew version of
# mpich requires gcc, and we might be building with clang.
#brew install mpich

# Install MPICH with our serial compiler.  Needed to install
# mpi4py later for testing (MPI is not bundled with the wheel).
# NOTE:  on arm64 we cannot run tests anyway (since we are
# cross compiling).  So do not attempt to build mpich / mpi4py.

if [ "${arch}" != "macosx_arm64" ] && [ "x${quick}" = "x" ]; then
    mpich_version=3.4
    mpich_dir=mpich-${mpich_version}
    mpich_pkg=${mpich_dir}.tar.gz

    echo "Fetching MPICH..."

    if [ ! -e ${mpich_pkg} ]; then
        curl -SL https://www.mpich.org/static/downloads/${mpich_version}/${mpich_pkg} -o ${mpich_pkg}
    fi

    echo "Building MPICH..."

    rm -rf ${mpich_dir}
    fcopt="--enable-fortran=all"
    if [ "x${FC}" = "x" ]; then
        fcopt="--disable-fortran"
    fi
    unset F90
    unset F90FLAGS
    tar xzf ${mpich_pkg} \
        && cd ${mpich_dir} \
        && CC="${CC}" CXX="${CXX}" FC="${FC}" F77="${FC}" \
        CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" \
        FFLAGS="${FCFLAGS}" FCFLAGS="${FCFLAGS}" \
        MPICH_MPICC_CFLAGS="${CFLAGS}" \
        MPICH_MPICXX_CXXFLAGS="${CXXFLAGS}" \
        MPICH_MPIF77_FFLAGS="${FCFLAGS}" \
        MPICH_MPIFORT_FCFLAGS="${FCFLAGS}" \
        ./configure ${fcopt} \
        --with-device=ch3 \
        --prefix="${PREFIX}" \
        && make -j ${MAKEJ} \
        && make install
fi

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

# Update pip
pip install --upgrade pip

# Install a couple of base packages that are always required
pip install -v "numpy<${numpy_ver}" cmake wheel

# Install build requirements.
CC="${CC}" CFLAGS="${CFLAGS}" pip install -v -r "${scriptdir}/build_requirements.txt"

# Install mpi4py for running tests (only on x86_64)

if [ "${arch}" != "macosx_arm64" ] && [ "x${quick}" = "x" ]; then
    echo "mpicc = $(which mpicc)"
    echo "mpicxx = $(which mpicxx)"
    echo "mpirun = $(which mpirun)"
    pip install mpi4py
fi

# Optionally Install Openblas

if [ "x${use_gcc}" = "xyes" ]; then
    openblas_version=0.3.20
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
        CC="${CC}" FC="${FC}" DYNAMIC_ARCH=1 TARGET=GENERIC \
        COMMON_OPT="${CFLAGS}" FCOMMON_OPT="${FCFLAGS}" \
        LDFLAGS="-fopenmp -lm" libs netlib shared \
        && make NO_SHARED=1 DYNAMIC_ARCH=1 TARGET=GENERIC PREFIX="${PREFIX}" install \
        && popd >/dev/null 2>&1
fi

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
    --enable-static \
    --disable-shared \
    --prefix="${PREFIX}" \
    && make -j ${MAKEJ} \
    && make install \
    && popd >/dev/null 2>&1

if [ "x${quick}" = "x" ]; then
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
        ./configure ${cross} \
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
        ./configure ${cross} \
        --enable-static \
        --disable-shared \
        --with-pic \
        --with-gmp="${PREFIX}" \
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

    ssparse_version=5.11.0
    ssparse_dir=SuiteSparse-${ssparse_version}
    ssparse_pkg=${ssparse_dir}.tar.gz

    echo "Fetching SuiteSparse..."

    if [ ! -e ${ssparse_pkg} ]; then
        curl -SL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v${ssparse_version}.tar.gz -o ${ssparse_pkg}
    fi

    echo "Building SuiteSparse..."

    blas_opt="-framework Accelerate"
    if [ "x${use_gcc}" = "xyes" ]; then
        blas_opt="-lopenblas -fopenmp -lm"
    fi

    rm -rf ${ssparse_dir}
    tar xzf ${ssparse_pkg} \
        && pushd ${ssparse_dir} >/dev/null 2>&1 \
        && patch -p1 < "${scriptdir}/suitesparse.patch" \
        && make library JOBS=${MAKEJ} \
        CC="${CC}" CXX="${CXX}" \
        CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" AUTOCC=no \
        GPU_CONFIG="" BLAS="${blas_opt}" \
        && make static JOBS=${MAKEJ} \
        CC="${CC}" CXX="${CXX}" \
        CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" AUTOCC=no \
        GPU_CONFIG="" BLAS="${blas_opt}" \
        && cp -a ./include/* "${PREFIX}/include/" \
        && find . -name "*.a" -exec cp -a '{}' "${PREFIX}/lib/" \; \
        && popd >/dev/null 2>&1

    # This line removed from above, since we are linking to static libs:
    # && cp -a ./lib/* "${PREFIX}/lib/" \
fi
