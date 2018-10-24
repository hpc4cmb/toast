#!/bin/bash

# WARNING:  If you upgrade versions of python3 (i.e. 3.6 --> 3.7) you will
# need to rerun this install.

# NOTE:  This script expects to be located in the toast/external/static
# directory, so that it can find patch files at the relative location.  You
# can run the script in some other working directory, but leave this file
# in place.  For example:
#
# $>  cd workdir
# $>  /path/to/toast/external/static/install_ubuntu_16.04.sh | tee log
#

# ---------------------- Install options ------------------------

# Set this to the install prefix for compiled dependencies.  If only the root
# user can write to this directory, then you will have to run this script with
# "sudo".
PREFIX=/home/kisner/software/toastdeps

# Set this to the unix group that should own the final installed software.
# If empty, it will not be changed.
CHGRP=""

# Set this to the permissions string that should be used on the final installed
# software.  If empty, it will not be changed.  This default ensures that no
# one can change the installed files without first restoring write permission.
CHMOD=""


# ------------- Install dependencies provided by the OS. ---------------
#
# Use APT to get everything we can.  Alternatively, you could provide the
# python packages here by installing Anaconda.
#

sudo apt-get update

sudo apt-get install -y curl procps build-essential gfortran git subversion \
    autoconf automake libtool m4 cmake locales libgl1-mesa-glx xvfb \
    libfftw3-dev \
    libopenblas-dev \
    libcfitsio-dev \
    libhdf5-dev \
    python3 \
    libpython3-dev \
    python3-pip \
    python3-nose \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-yaml \
    python3-h5py \
    cython3

# ------------------- Set up install prefix ------------------------
#
# Much of this stuff is not strictly necessary- it is just making the
# software nicer to load into user shell environments.
#

# Create the prefix directories

mkdir -p "${PREFIX}/lib"
mkdir -p "${PREFIX}/bin"
mkdir -p "${PREFIX}/include"
mkdir -p "${PREFIX}/etc"
pushd "${PREFIX}"
if [ ! -e lib64 ]; then
    ln -s lib lib64
fi
popd > /dev/null

# In the ${PREFIX}/etc directory above, create a small shell file which will
# load all this software into a user's environment.

ENV_FILE="${PREFIX}/etc/toast_env.sh"

# What is the name of our python site-packages directory?
PYSITE=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
mkdir -p "${PREFIX}/lib/python${PYSITE}/site-packages"

echo "# Load TOAST dependencies into the environment" > "${ENV_FILE}"

echo "export PATH=\"${PREFIX}/bin:${PATH}\"" >> "${ENV_FILE}"
echo "export CPATH=\"${PREFIX}/include:${CPATH}\"" >> "${ENV_FILE}"
echo "export LIBRARY_PATH=\"${PREFIX}/lib:${LIBRARY_PATH}\"" >> "${ENV_FILE}"
echo "export LD_LIBRARY_PATH=\"${PREFIX}/lib:${LD_LIBRARY_PATH}\"" >> "${ENV_FILE}"
echo "export PYTHONPATH=\"${PREFIX}/lib/python${PYSITE}/site-packages:${PYTHONPATH}\"" >> "${ENV_FILE}"
echo "" >> "${ENV_FILE}"

# Load this shell snippet now, to put the prefix into our search paths

. "${ENV_FILE}"


# -------------- Install other required software  --------------------
#
# These are required packages where we would really like to have the
# latest stable versions (i.e. not from the OS package manager).
#

# Use pip where possible to install packages into our separate prefix.
# Note that this will pull in other updated dependencies, but we are
# putting this into our own prefix (not /usr), so it is fine.  One could
# also use a virtualenv or install these via anaconda if you were using
# anaconda already to provide python3.

pip3 install --no-cache-dir --system --target="${PREFIX}/lib/python${PYSITE}/site-packages" healpy

pip3 install --no-cache-dir --system --target="${PREFIX}/lib/python${PYSITE}/site-packages" ephem

pip3 install --no-cache-dir --system --target="${PREFIX}/lib/python${PYSITE}/site-packages" https://github.com/bthorne93/PySM_public/archive/2.1.0.tar.gz

# Install our own MPICH

curl -SL http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
    | tar -xzf - \
    && cd mpich-3.2 \
    && CC="gcc" CXX="g++" FC="gfortran" \
    CFLAGS="-O3 -fPIC -pthread" CXXFLAGS="-O3 -fPIC -pthread" FCFLAGS="-O3 -fPIC -pthread" \
    ./configure --prefix="${PREFIX}" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf mpich-3.2*

# Install Latest mpi4py.  We want to build this ourselves so that it uses the
# MPI version we have installed (MPICH or OpenMPI).  We also need a version
# >= 2.0 so that we can pass the communicator between python and C++.

 curl -SL https://pypi.python.org/packages/31/27/1288918ac230cc9abc0da17d84d66f3db477757d90b3d6b070d709391a15/mpi4py-3.0.0.tar.gz#md5=bfe19f20cef5e92f6e49e50fb627ee70 \
    | tar xzf - \
    && cd mpi4py-3.0.0 \
    && python3 setup.py install --prefix="${PREFIX}" \
    && cd .. \
    && rm -rf mpi4py*


#--------------------- Optional dependecies ---------------------
#
# Although not strictly required, these packages enable important
# features, such as destriping map-making, 4Pi beam convolution,
# atmosphere simulation and bandpass integration.
#

# Where is this script located, so we can find any patch files?

pushd $(dirname $0) > /dev/null
SDIR=$(pwd)
popd > /dev/null

# Install latest libmadam

 curl -SL https://github.com/hpc4cmb/libmadam/releases/download/0.2.7/libmadam-0.2.7.tar.bz2 \
    | tar -xjf - \
    && cd libmadam-0.2.7 \
    && FC="mpifort" MPIFC="mpifort" FCFLAGS="-O3 -fPIC -pthread" \
    CC="mpicc" MPICC="mpicc" CFLAGS="-O3 -fPIC -pthread" \
    ./configure  --with-cfitsio="/usr" \
    --with-blas="-lopenblas" --with-lapack="" \
    --with-fftw="/usr" --prefix="${PREFIX}" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf libmadam*

# Install conviqt

 curl -SL https://www.dropbox.com/s/4tzjn9bgq7enkf9/libconviqt-1.0.2.tar.bz2?dl=0 \
    | tar -xjf - \
    && cd libconviqt-1.0.2 \
    && CC="mpicc" CXX="mpicxx" MPICC="mpicc" MPICXX="mpicxx" \
    CFLAGS="-O3 -fPIC -pthread -std=gnu99" CXXFLAGS="-O3 -fPIC -pthread" \
    ./configure  --with-cfitsio="/usr" --prefix="${PREFIX}" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf libconviqt*

# Install elemental

 curl -SL https://github.com/elemental/Elemental/archive/v0.87.7.tar.gz \
    | tar -xzf - \
    && cd Elemental-0.87.7 \
    && mkdir build && cd build \
    && cmake \
    -D CMAKE_INSTALL_PREFIX="${PREFIX}" \
    -D INSTALL_PYTHON_PACKAGE=OFF \
    -D CMAKE_CXX_COMPILER="mpicxx" \
    -D CMAKE_C_COMPILER="mpicc" \
    -D CMAKE_Fortran_COMPILER="mpifort" \
    -D MPI_CXX_COMPILER="mpicxx" \
    -D MPI_C_COMPILER="mpicc" \
    -D MPI_Fortran_COMPILER="mpifort" \
    -D METIS_GKREGEX=ON \
    -D EL_DISABLE_PARMETIS=TRUE \
    -D MATH_LIBS="$(echo ' -lopenblas' | sed -e 's#^ ##g')" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_FLAGS="-O3 -fPIC -pthread -fopenmp" \
    -D CMAKE_C_FLAGS="-O3 -fPIC -pthread -fopenmp" \
    -D CMAKE_Fortran_FLAGS="-O3 -fPIC -pthread -fopenmp" \
    -D CMAKE_SHARED_LINKER_FLAGS="-fopenmp" \
    .. \
    && make -j 4 && make install \
    && cd ../.. \
    && rm -rf Elemental*

# Install libsharp

 git clone https://github.com/Libsharp/libsharp --branch master --single-branch --depth 1 \
    && cd libsharp \
    && patch -p1 < "${SDIR}/../rules/patch_libsharp" \
    && autoreconf \
    && CC="mpicc" CFLAGS="-O3 -fPIC -pthread" \
    ./configure  --enable-mpi --enable-pic --prefix="${PREFIX}" \
    && make \
    && cp -a auto/* "${PREFIX}" \
    && cd python \
    && LIBSHARP="${PREFIX}" CC="mpicc -g" LDSHARED="mpicc -g -shared" \
    python3 setup.py install --prefix="${PREFIX}" \
    && cd ../.. \
    && rm -rf libsharp*

# Install aatm

 curl -SL https://launchpad.net/aatm/trunk/0.5/+download/aatm-0.5.tar.gz \
    | tar xzf - \
    && cd aatm-0.5 \
    && chmod -R u+w . \
    && patch -p1 < "${SDIR}/../rules/patch_aatm" \
    && autoreconf \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" \
    CPPFLAGS="-I${PREFIX}/include" \
    LDFLAGS="-L${PREFIX}/lib" \
    ./configure  \
    --prefix="${PREFIX}" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf aatm*

# The following commands to install libbz2 and boost are needed if you
# plan on building the spt3g_software package.  Otherwise you can comment
# them out.

# Install libbz2

 curl -SL https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/bzip2/1.0.6-8/bzip2_1.0.6.orig.tar.bz2 \
    | tar xjf - \
    && cd bzip2-1.0.6 \
    && patch -p1 < "${SDIR}/../rules/patch_bzip2" \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" \
    make -f Makefile-toast \
    && cp -a bzlib.h "${PREFIX}/include" \
    && cp -a libbz2.so* "${PREFIX}/lib" \
    && cd .. \
    && rm -rf bzip2*

# Install Boost

 curl -SL https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.bz2 \
    -o boost_1_65_1.tar.bz2 \
    && tar xjf boost_1_65_1.tar.bz2 \
    && cd boost_1_65_1 \
    && echo "" > tools/build/user-config.jam \
    && echo "using mpi : mpicxx : <include>\"${PREFIX}/include\" <library-path>\"${PREFIX}/lib\" <find-shared-library>\"mpicxx\" <find-shared-library>\"mpi\" ;" >> tools/build/user-config.jam \
    && echo "option jobs : 4 ;" >> tools/build/user-config.jam \
    && BOOST_BUILD_USER_CONFIG=tools/build/user-config.jam \
    BZIP2_INCLUDE="${PREFIX}/include" \
    BZIP2_LIBPATH="${PREFIX}/lib" \
    ./bootstrap.sh \
    --with-toolset=gcc \
    --with-python=python3.6 \
    --prefix="${PREFIX}" \
    && ./b2 --layout=tagged --user-config=./tools/build/user-config.jam\
    $(python3.6-config --includes | sed -e 's/-I//g' -e 's/\([^[:space:]]\+\)/ include=\1/g') \
    variant=release threading=multi link=shared runtime-link=shared install \
    && cd .. \
    && rm -rf boost*

# Make sure that any python modules installed to our prefix are built into pyc
# files- so that we can make those directories read-only

python3 -m compileall -f "${PREFIX}/lib/python${PYSITE}/site-packages"

# Set permissions

if [ "x${CHGRP}" != "x" ]; then
    chgrp -R ${CHGRP} "${PREFIX}"
fi

if [ "x${CHMOD}" != "x" ]; then
    chmod -R ${CHMOD} "${PREFIX}"
fi
