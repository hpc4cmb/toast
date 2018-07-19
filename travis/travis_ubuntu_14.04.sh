#!/bin/bash
#
# This script is designed to be run *INSIDE* a docker container that has
# been launched like:
#
#    %> ./travis_docker.sh <instance ID>
#
# Once you are inside the docker container, you can then run this script with
#
#    %> ./scripts/travis_ubuntu_14.04.sh <gcc suffix>
#
# Where "gcc suffix" is something like "-7", "-6", "-5", or "".  This will
# build dependencies and create a tarball.  Then upload the tarball with:
#
#    %> ./scripts/travis_upload.sh <NERSC user ID>
#
# You must be in the "cmb" filegroup for this to work.
#
# Now exit the container and *REPEAT* this entire list of instructions
# (running a new container) for every gcc version in the build matrix.
#

usage () {
    echo "$0 <toolchain suffix>"
    exit 1
}

# The first argument to this script should select the package extension
# for the compilers, like "", "-5", "-6", "-7".
toolchain="$1"

# Runtime packages.  These are packages that will be installed into system
# locations when travis is actually running.  We install them here so that we
# can use them when building our cached dependencies.

# Install APT packages
sudo -E apt-add-repository -y "ppa:ubuntu-toolchain-r/test"
sudo -E apt-get -yq update
sudo -E apt-get -yq --no-install-suggests --no-install-recommends install build-essential git autoconf automake m4 libtool cmake pkg-config locales libgl1-mesa-glx xvfb libopenblas-dev liblapack-dev libfftw3-dev libhdf5-dev libcfitsio3-dev gcc${toolchain} g++${toolchain} gfortran${toolchain}

# Export serial compiler variables
export CC=$(which gcc${toolchain})
export CXX=$(which g++${toolchain})
export FC=$(which gfortran${toolchain})

# Install travis python
if [ ! -e "${HOME}/virtualenv/python3.6/bin/activate" ]; then
    wget https://s3.amazonaws.com/travis-python-archives/binaries/ubuntu/14.04/x86_64/python-3.6.tar.bz2
    sudo tar xjf python-3.6.tar.bz2 --directory /
fi
source ${HOME}/virtualenv/python3.6/bin/activate

# Pip install the runtime packages we need.  These are not saved in the tarball.
pip install numpy scipy matplotlib cython astropy ephem healpy

# Set up TOAST dependencies for travis.  We use the following approach:
#
# Python     : Use the travis-provided virtualenv.  Pip install common packages
#              like numpy, scipy, astropy, etc.
# MPI        : Install MPICH from source using the serial compilers specified
#              in our build matrix.
# mpi4py     : Install from source to use our MPICH.
# ephem      : Install with pip using travis virtualenv.
# healpy     : Install with pip using travis virtualenv.
# elemental  : Install using our MPICH and build matrix compilers.
# aatm       : Install with our build matrix compilers.
# libmadam   : Install using our MPICH and build matrix compilers.
# libconviqt : Install using our MPICH and build matrix compilers.
# libsharp   : Install using our MPICH and build matrix compilers.
# PySM       : Install with pip using travis virtualenv.
#

PREFIX="${HOME}/software"
mkdir -p "${PREFIX}"

# Serial compilers - these should be set in the build matrix

echo "Building dependencies with:"
echo "  CC = ${CC} $(${CC} -dumpversion)"
echo "  CXX = ${CXX} $(${CXX} -dumpversion)"
echo "  FC = ${FC} $(${FC} -dumpversion)"

# Set up our install prefix for all manually installed software

mkdir -p ${PREFIX}/bin
mkdir -p ${PREFIX}/lib
mkdir -p ${PREFIX}/include
export PYSITE=$(python --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
mkdir -p ${PREFIX}/lib/python${PYSITE}/site-packages
export PATH="${PREFIX}/bin:${PATH}"
export CPATH="${PREFIX}/include:${CPATH}"
export LIBRARY_PATH="${PREFIX}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${PREFIX}/lib/python${PYSITE}/site-packages:${PYTHONPATH}"

# Install MPICH

wget http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
    && tar -xzf mpich-3.2.tar.gz \
    && cd mpich-3.2 \
    && CFLAGS="-O2 -g -fPIC -pthread" \
    CXXFLAGS="-O2 -g -fPIC -pthread" \
    FCFLAGS="-O2 -g -fPIC -pthread" \
    ./configure --prefix="${PREFIX}" \
    && make \
    && make install \
    && cd ..

# set the MPI compilers
export MPICC=$(which mpicc)
export MPICXX=$(which mpicxx)
export MPIFC=$(which mpif90)

# Install mpi4py using our MPICH.  This is put into the tarball.

pip install --target="${PREFIX}/lib/python${PYSITE}/site-packages" mpi4py \

# Install aatm

wget https://launchpad.net/aatm/trunk/0.5/+download/aatm-0.5.tar.gz \
    && tar xzf aatm-0.5.tar.gz \
    && wget https://raw.githubusercontent.com/hpc4cmb/toast/master/external/rules/patch_aatm \
    && cd aatm-0.5 \
    && chmod -R u+w . \
    && patch -p1 < "../patch_aatm" \
    && autoreconf --force --install \
    && CFLAGS="-O2 -g -fPIC -pthread" \
    CPPFLAGS="-I${PREFIX}/include" \
    LDFLAGS="-L${PREFIX}/lib" \
    ./configure  \
    --prefix="${PREFIX}" \
    && make \
    && make install \
    && cd ..

# Install Elemental

wget https://github.com/elemental/Elemental/archive/v0.87.7.tar.gz \
    && tar -xzf v0.87.7.tar.gz \
    && cd Elemental-0.87.7 \
    && mkdir build && cd build \
    && cmake \
    -D CMAKE_INSTALL_PREFIX="${PREFIX}" \
    -D INSTALL_PYTHON_PACKAGE=OFF \
    -D CMAKE_CXX_COMPILER="${MPICXX}" \
    -D CMAKE_C_COMPILER="${MPICC}" \
    -D CMAKE_Fortran_COMPILER="${MPIFC}" \
    -D MPI_CXX_COMPILER="${MPICXX}" \
    -D MPI_C_COMPILER="${MPICC}" \
    -D MPI_Fortran_COMPILER="${MPIFC}" \
    -D METIS_GKREGEX=ON \
    -D EL_DISABLE_PARMETIS=TRUE \
    -D MATH_LIBS="-llapack -lopenblas" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_FLAGS="-O2 -g -fPIC -pthread -fopenmp" \
    -D CMAKE_C_FLAGS="-O2 -g -fPIC -pthread -fopenmp" \
    -D CMAKE_Fortran_FLAGS="-O2 -g -fPIC -pthread -fopenmp" \
    -D CMAKE_SHARED_LINKER_FLAGS="-fopenmp" \
    .. \
    && make \
    && make install \
    && cd ../..

# Install libmadam

wget https://github.com/hpc4cmb/libmadam/releases/download/0.2.7/libmadam-0.2.7.tar.bz2 \
    && tar -xjf libmadam-0.2.7.tar.bz2 \
    && cd libmadam-0.2.7 \
    && FC="${MPIFC}" FCFLAGS="-O2 -g -fPIC -pthread" \
    CC="${MPICC}" CFLAGS="-O2 -g -fPIC -pthread" \
    ./configure --with-cfitsio="/usr" \
    --with-blas="-lopenblas" --with-lapack="-llapack" \
    --with-fftw="/usr" --prefix="${PREFIX}" \
    && make \
    && make install \
    && cd ..

# Install libconviqt

wget -O libconviqt-1.0.2.tar.bz2 https://www.dropbox.com/s/4tzjn9bgq7enkf9/libconviqt-1.0.2.tar.bz2?dl=1 \
    && tar -xjf libconviqt-1.0.2.tar.bz2 \
    && cd libconviqt-1.0.2 \
    && CC="${MPICC}" CXX="${MPICXX}" \
    CFLAGS="-O2 -g -fPIC -pthread -std=gnu99" \
    CXXFLAGS="-O2 -g -fPIC -pthread" \
    ./configure --with-cfitsio="/usr" \
    --prefix="${PREFIX}" \
    && make \
    && make install \
    && cd ..

# Install libsharp

git clone https://github.com/Libsharp/libsharp --branch master --single-branch --depth 1 \
    && wget https://raw.githubusercontent.com/hpc4cmb/toast/master/external/rules/patch_libsharp \
    && cd libsharp \
    && patch -p1 < "../patch_libsharp" \
    && autoreconf --force --install \
    && CC="${MPICC}" CFLAGS="-O2 -g -fPIC -pthread -std=c99" \
    ./configure  --enable-mpi --enable-pic --prefix="${PREFIX}" \
    && make \
    && cp -a auto/* "${PREFIX}" \
    && cd python \
    && LIBSHARP="${PREFIX}" CC="${MPICC} -g" LDSHARED="${MPICC} -g -shared" \
    python setup.py install --prefix="${PREFIX}" \
    && cd ../..

# Make sure that any python modules installed to our prefix are built into pyc
# files- so that we can make those directories read-only

python -m compileall -f "${PREFIX}/lib/python${PYSITE}/site-packages"

# Package up tarball
cd "${HOME}" \
    && export \
    TARBALL="travis_14.04_gcc${toolchain}_python${PYSITE}.tar.bz2" \
    && tar cjvf "${TARBALL}" software
