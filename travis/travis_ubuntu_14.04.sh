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
toolchainver=$(echo $toolchain | sed -e "s/^-//")

# Runtime packages.  These are packages that will be installed into system
# locations when travis is actually running.  We install them here so that we
# can use them when building our cached dependencies.

# Install APT packages
sudo -E apt-add-repository -y "ppa:ubuntu-toolchain-r/test"
sudo -E apt-get -yq update
sudo -E apt-get -yq --no-install-suggests --no-install-recommends install build-essential git autoconf automake m4 libtool pkg-config locales libgl1-mesa-glx xvfb libopenblas-dev liblapack-dev libfftw3-dev libhdf5-dev libcfitsio3-dev gcc${toolchain} g++${toolchain} gfortran${toolchain}

# Export serial compiler variables
export CC=$(which gcc${toolchain})
export CXX=$(which g++${toolchain})
export FC=$(which gfortran${toolchain})

# Install travis python
# if [ ! -e "${HOME}/virtualenv/python3.6/bin/activate" ]; then
#     wget https://s3.amazonaws.com/travis-python-archives/binaries/ubuntu/14.04/x86_64/python-3.6.tar.bz2
#     sudo tar xjf python-3.6.tar.bz2 --directory /
# fi
source ${HOME}/virtualenv/python3.6/bin/activate

# Pip install the runtime packages we need.  These are not saved in the tarball.
pip install numpy scipy matplotlib cython astropy ephem healpy cmake

# Set up TOAST dependencies for travis.  We use the following approach:
#
# Python     : Use the travis-provided virtualenv.  Pip install common packages
#              like numpy, scipy, astropy, etc.
# MPI        : Install MPICH from source using the serial compilers specified
#              in our build matrix.
# mpi4py     : Install from source to use our MPICH.
# ephem      : Install with pip using travis virtualenv.
# healpy     : Install with pip using travis virtualenv.
# suitesparse: Install using our build matrix compilers.
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

# libbz2, needed for boost / spt3g
# FIXME: change patch link below after branch is merged upstream.

curl -SL https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/bzip2/1.0.6-8/bzip2_1.0.6.orig.tar.bz2 \
        | tar xjf - \
        && wget https://raw.githubusercontent.com/tskisner/toast/issue_235/external/rules/patch_bzip2 \
        && cd bzip2-1.0.6 \
        && patch -p1 < ../patch_bzip2 \
        && CC="${CC}" CFLAGS="-O2 -g -fPIC -pthread" \
        make -f Makefile-toast \
        && cp -a bzlib.h "${PREFIX}/include" \
        && cp -a libbz2.so* "${PREFIX}/lib" \
        && cd ..

# Install boost (needed by spt3g)

curl -SL https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.bz2 \
    -o boost_1_65_1.tar.bz2 \
    && tar xjf boost_1_65_1.tar.bz2 \
    && cd boost_1_65_1 \
    && echo "" > tools/build/src/user-config.jam \
    && echo "using gcc : ${toolchainver} : ${CXX} ;" >> tools/build/src/user-config.jam \
    && echo "using mpi : ${MPICXX} : <include>\"${PREFIX}/include\" <library-path>\"${PREFIX}/lib\" <find-shared-library>\"mpichcxx\" <find-shared-library>\"mpich\" ;" >> tools/build/src/user-config.jam \
    && BOOST_BUILD_USER_CONFIG=tools/build/src/user-config.jam \
    BZIP2_INCLUDE="${PREFIX}/include" \
    BZIP2_LIBPATH="${PREFIX}/lib" \
    ./bootstrap.sh \
    --with-toolset=gcc \
    --with-python=python${PYSITE} \
    --prefix=${PREFIX} \
    && ./b2 --toolset=gcc${toolchain} --layout=tagged \
    --user-config=./tools/build/src/user-config.jam \
    $(python3-config --includes | sed -e 's/-I//g' -e 's/\([^[:space:]]\+\)/ include=\1/g') \
    variant=release threading=multi link=shared runtime-link=shared install \
    && cd ..

# Install SuiteSparse

curl -SL http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-5.4.0.tar.gz \
    -o SuiteSparse-5.4.0.tar.gz \
    && tar xzf SuiteSparse-5.4.0.tar.gz \
    && cd SuiteSparse \
    && make CC="${CC}" CXX="${CXX}" CFLAGS="-O2 -g -fPIC -pthread" AUTOCC=no \
    F77="${FC}" F77FLAGS="-O2 -g -fPIC -pthread" \
    CFOPENMP="-fopenmp" LAPACK="-llapack" BLAS="-lopenblas" \
    && make install CC="gcc" CXX="g++" \
    CFLAGS="-O2 -g -fPIC -pthread" AUTOCC=no \
    F77="gfortran" F77FLAGS="-O2 -g -fPIC -pthread" \
    CFOPENMP="-fopenmp" LAPACK="-llapack" BLAS="-lopenblas" \
    INSTALL="${PREFIX}" \
    && cd ..

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

# Install spt3g software
# FIXME: change patch link below after branch is merged upstream.

git clone https://github.com/CMB-S4/spt3g_software.git --branch master --single-branch --depth 1 \
    && wget https://raw.githubusercontent.com/tskisner/toast/issue_235/external/rules/patch_spt3g \
    && export spt3g_start=$(pwd) \
    && cd spt3g_software \
    && patch -p1 < ../patch_spt3g \
    && cd .. \
    && cp -a spt3g_software "${PREFIX}/spt3g" \
    && cd "${PREFIX}/spt3g" \
    && mkdir build \
    && cd build \
    && LDFLAGS="-Wl,-z,muldefs" \
    cmake \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="-O2 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O2 -g -fPIC -pthread" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DBOOST_ROOT="${PREFIX}" \
    .. \
    && make \
    && ln -s ${PREFIX}/spt3g/build/bin/* ${PREFIX}/bin/ \
    && ln -s ${PREFIX}/spt3g/build/spt3g ${PREFIX}/lib/python${PYSITE}/site-packages/ \
    && cd ${spt3g_start}

# Install TIDAS

git clone https://github.com/hpc4cmb/tidas.git --branch master --single-branch --depth 1 \
    && cd tidas \
    && mkdir build \
    && cd build \
    && cmake \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DMPI_C_COMPILER="${MPICC}" \
    -DMPI_CXX_COMPILER="${MPICXX}" \
    -DCMAKE_C_FLAGS="-O2 -g -fPIC -pthread -DSQLITE_DISABLE_INTRINSIC" \
    -DCMAKE_CXX_FLAGS="-O2 -g -fPIC -pthread -DSQLITE_DISABLE_INTRINSIC" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    .. \
    && make && make install \
    && cd ../..

# Make sure that any python modules installed to our prefix are built into pyc
# files- so that we can make those directories read-only

python -m compileall -f "${PREFIX}/lib/python${PYSITE}/site-packages"

# Package up tarball
cd "${HOME}" \
    && export \
    TARBALL="travis_14.04_gcc${toolchain}_python${PYSITE}.tar.bz2" \
    && tar cjf "${TARBALL}" software
