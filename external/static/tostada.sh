#!/bin/bash

# Change these as needed

PREFIX="/opt/toast"
VERSION=$(date "+%Y%m%d")

PREFIX_AUX="${PREFIX}/deps/${VERSION}/aux"
PREFIX_CONDA="${PREFIX}/deps/${VERSION}/conda"

# Install dependencies provided by the OS.  This is only needed once, so
# comment it out after the first time.

#sudo apt-get update

#sudo apt-get install -y curl procps build-essential gfortran git subversion \
#    autoconf automake libtool m4 cmake locales libgl1-mesa-glx xvfb \
#    libopenblas-dev libcfitsio-dev \

# Install TOAST dependencies from source.

mkdir -p ${PREFIX_CONDA}/bin
mkdir -p ${PREFIX_CONDA}/lib
mkdir -p ${PREFIX_AUX}/lib/python3.6/site-packages
pushd ${PREFIX_AUX}
if [ ! -e lib64 ]; then
    ln -s lib lib64
fi
popd > /dev/null

export TOAST_AUX_ROOT=${PREFIX_AUX}
export CMAKE_PREFIX_PATH=${PREFIX_AUX}:${CMAKE_PREFIX_PATH}
export PATH=${PREFIX_AUX}/bin:${PREFIX_CONDA}/bin:${PATH}
export CPATH=${PREFIX_AUX}/include:${CPATH}
export LIBRARY_PATH=${PREFIX_AUX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${PREFIX_AUX}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${PREFIX_AUX}/lib/python3.6/site-packages:${PYTHONPATH}

# Install conda root environment.  Symlink libpython development files into
# the auxiliary prefix so that it is in LD_LIBRARY_PATH for future linking.

 curl -SL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o miniconda.sh \
    && /bin/bash miniconda.sh -b -f -p ${PREFIX_CONDA} \
    && conda install --copy --yes python=3.6 \
    && rm miniconda.sh \
    && rm -rf ${PREFIX_CONDA}/pkgs/* \
    && ln -s ${PREFIX_CONDA}/lib/libpython* ${PREFIX_AUX}/lib/

# Install conda packages.

 conda install --copy --yes \
    nose \
    cython \
    numpy \
    scipy \
    matplotlib \
    pyyaml \
    astropy \
    psutil \
    ephem \
    virtualenv \
    pandas \
    memory_profiler \
    ipython \
    && python -c "import matplotlib.font_manager" \
    && rm -rf ${PREFIX_CONDA}/pkgs/*

conda list --export | grep -v conda > "${PREFIX_CONDA}/pkg_list.txt"

# Install pip packages.

 pip install --no-binary :all: \
    fitsio \
    timemory \
    healpy

# Where is this script located, so we can find any patch files?

pushd $(dirname $0) > /dev/null
SDIR=$(pwd)
popd > /dev/null

# Install our own MPICH

curl -SL http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
    | tar -xzf - \
    && cd mpich-3.2 \
    && CC="gcc" CXX="g++" FC="gfortran" \
    CFLAGS="-O3 -fPIC -pthread" CXXFLAGS="-O3 -fPIC -pthread" FCFLAGS="-O3 -fPIC -pthread" \
    ./configure --prefix="${PREFIX_AUX}" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf mpich-3.2*

# Install mpi4py.

 curl -SL https://pypi.python.org/packages/31/27/1288918ac230cc9abc0da17d84d66f3db477757d90b3d6b070d709391a15/mpi4py-3.0.0.tar.gz#md5=bfe19f20cef5e92f6e49e50fb627ee70 \
    | tar xzf - \
    && cd mpi4py-3.0.0 \
    && python3 setup.py install --prefix="${PREFIX_AUX}" \
    && cd .. \
    && rm -rf mpi4py*

# Install wcslib

 curl -SL ftp://ftp.atnf.csiro.au/pub/software/wcslib/wcslib-5.18.tar.bz2 \
    | tar xjf - \
    && cd wcslib-5.18 \
    && chmod -R u+w . \
    && patch -p1 < "${SDIR}/../rules/patch_wcslib" \
    && autoconf \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" \
    CPPFLAGS="-I${PREFIX_AUX}/include" \
    LDFLAGS="-L${PREFIX_AUX}/lib" \
    ./configure  \
    --disable-fortran \
    --prefix="${PREFIX_AUX}" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf wcslib*

# Install FFTW.

 curl -SL http://www.fftw.org/fftw-3.3.7.tar.gz \
    | tar xzf - \
    && cd fftw-3.3.7 \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" ./configure --enable-threads  --prefix="${PREFIX_AUX}" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf fftw*

# Install libbz2

 curl -SL https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/bzip2/1.0.6-8/bzip2_1.0.6.orig.tar.bz2 \
    | tar xjf - \
    && cd bzip2-1.0.6 \
    && patch -p1 < "${SDIR}/../rules/patch_bzip2" \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" \
    make -f Makefile-toast \
    && cp -a bzlib.h "${PREFIX_AUX}/include" \
    && cp -a libbz2.so* "${PREFIX_AUX}/lib" \
    && cd .. \
    && rm -rf bzip2*

# Install Boost

 curl -SL https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.bz2 \
    -o boost_1_65_1.tar.bz2 \
    && tar xjf boost_1_65_1.tar.bz2 \
    && cd boost_1_65_1 \
    && echo "" > tools/build/user-config.jam \
    && echo "using mpi : mpicxx : <include>\"/usr/include/mpi\" <library-path>\"/usr/lib\" <find-shared-library>\"mpi++\" <find-shared-library>\"mpi\" ;" >> tools/build/user-config.jam \
    && echo "option jobs : 4 ;" >> tools/build/user-config.jam \
    && BOOST_BUILD_USER_CONFIG=tools/build/user-config.jam \
    BZIP2_INCLUDE="${PREFIX_AUX}/include" \
    BZIP2_LIBPATH="${PREFIX_AUX}/lib" \
    ./bootstrap.sh \
    --with-toolset=gcc \
    --with-python=python3.6 \
    --prefix=${PREFIX_AUX} \
    && ./b2 --layout=tagged --user-config=./tools/build/user-config.jam\
    $(python3.6-config --includes | sed -e 's/-I//g' -e 's/\([^[:space:]]\+\)/ include=\1/g') \
    variant=release threading=multi link=shared runtime-link=shared install \
    && cd .. \
    && rm -rf boost*

# Install elemental

 curl -SL https://github.com/elemental/Elemental/archive/v0.87.7.tar.gz \
    | tar -xzf - \
    && cd Elemental-0.87.7 \
    && mkdir build && cd build \
    && cmake \
    -D CMAKE_INSTALL_PREFIX="${PREFIX_AUX}" \
    -D INSTALL_PYTHON_PACKAGE=OFF \
    -D CMAKE_CXX_COMPILER="mpicxx" \
    -D CMAKE_C_COMPILER="mpicc" \
    -D CMAKE_Fortran_COMPILER="mpifort" \
    -D MPI_CXX_COMPILER="mpicxx" \
    -D MPI_C_COMPILER="mpicc" \
    -D MPI_Fortran_COMPILER="mpifort" \
    -D METIS_GKREGEX=ON \
    -D EL_DISABLE_PARMETIS=TRUE \
    -D MATH_LIBS="-lopenblas" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_FLAGS="-O3 -fPIC -pthread -fopenmp" \
    -D CMAKE_C_FLAGS="-O3 -fPIC -pthread -fopenmp" \
    -D CMAKE_Fortran_FLAGS="-O3 -fPIC -pthread -fopenmp" \
    -D CMAKE_SHARED_LINKER_FLAGS="-fopenmp" \
    .. \
    && make -j 4 && make install \
    && cd ../.. \
    && rm -rf Elemental*

# Install HDF5

 curl -SL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.20/src/hdf5-1.8.20.tar.bz2 \
    | tar xjf - \
    && cd hdf5-1.8.20 \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" \
    CXX="g++" CXXFLAGS="-O3 -fPIC -pthread" \
    FC="gfortran" FCFLAGS="-O3 -fPIC -pthread" \
    ./configure \
    --disable-silent-rules \
    --disable-parallel \
    --enable-cxx \
    --enable-fortran \
    --enable-fortran2003 \
    --prefix="${PREFIX_AUX}" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf hdf5*

# Install h5py

 curl -SL https://pypi.python.org/packages/41/7a/6048de44c62fc5e618178ef9888850c3773a9e4be249e5e673ebce0402ff/h5py-2.7.1.tar.gz#md5=da630aebe3ab9fa218ac405a218e95e0 \
    | tar xzf - \
    && cd h5py-2.7.1 \
    && CC="gcc" LDSHARED="gcc -shared" \
    python setup.py install --prefix=${PREFIX_AUX} \
    && cd .. \
    && rm -rf h5py*

# Install aatm

 curl -SL https://launchpad.net/aatm/trunk/0.5/+download/aatm-0.5.tar.gz \
    | tar xzf - \
    && cd aatm-0.5 \
    && chmod -R u+w . \
    && patch -p1 < "${SDIR}/../rules/patch_aatm" \
    && autoreconf \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" \
    CPPFLAGS="-I${PREFIX_AUX}/include" \
    LDFLAGS="-L${PREFIX_AUX}/lib" \
    ./configure  \
    --prefix="${PREFIX_AUX}" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf aatm*

# Install conviqt

 curl -SL https://www.dropbox.com/s/4tzjn9bgq7enkf9/libconviqt-1.0.2.tar.bz2?dl=0 \
    | tar -xjf - \
    && cd libconviqt-1.0.2 \
    && CC="mpicc" CXX="mpicxx" MPICC="mpicc" MPICXX="mpicxx" \
    CFLAGS="-O3 -fPIC -pthread -std=gnu99" CXXFLAGS="-O3 -fPIC -pthread" \
    ./configure  --with-cfitsio="${PREFIX_AUX}" --prefix="${PREFIX_AUX}" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf libconviqt*

# Install libsharp

 git clone https://github.com/Libsharp/libsharp --branch master --single-branch --depth 1 \
    && cd libsharp \
    && patch -p1 < "${SDIR}/../rules/patch_libsharp" \
    && autoreconf \
    && CC="mpicc" CFLAGS="-O3 -fPIC -pthread" \
    ./configure  --enable-mpi --enable-pic --prefix="${PREFIX_AUX}" \
    && make \
    && cp -a auto/* "${PREFIX_AUX}/" \
    && cd python \
    && LIBSHARP="${PREFIX_AUX}" CC="mpicc -g" LDSHARED="mpicc -g -shared" \
    python setup.py install --prefix="${PREFIX_AUX}" \
    && cd ../.. \
    && rm -rf libsharp*

# Install libmadam

 curl -SL https://github.com/hpc4cmb/libmadam/releases/download/0.2.9/libmadam-0.2.9.tar.bz2 \
    | tar -xjf - \
    && cd libmadam-0.2.9 \
    && FC="mpifort" MPIFC="mpifort" FCFLAGS="-O3 -fPIC -pthread" \
    CC="mpicc" MPICC="mpicc" CFLAGS="-O3 -fPIC -pthread" \
    ./configure  --with-cfitsio="${PREFIX_AUX}" \
    --with-blas="-lopenblas" --with-lapack="" \
    --with-fftw="${PREFIX_AUX}" --prefix="${PREFIX_AUX}" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf libmadam*

# Install SPT3G.  NOTE: this is using the S.A. specific spt3g fork.
# you should have keychain setup with your ssh key loaded so that this
# line does not ask for a password.

 git clone git@bitbucket.org:berkeleylab/spt3g_software_sa.git --branch master --single-branch --depth 1 \
    && export spt3g_start=$(pwd) \
    && cd spt3g_software_sa \
    && patch -p1 < "${SDIR}/../rules/patch_spt3g" \
    && cd .. \
    && cp -a spt3g_software_sa "${PREFIX_AUX}/spt3g" \
    && cd "${PREFIX_AUX}/spt3g" \
    && mkdir build \
    && cd build \
    && LDFLAGS="-Wl,-z,muldefs" \
    cmake \
    -DCMAKE_C_COMPILER="gcc" \
    -DCMAKE_CXX_COMPILER="g++" \
    -DCMAKE_C_FLAGS="-O3 -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -fPIC -pthread" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DPYTHON_EXECUTABLE:FILEPATH="${PREFIX_CONDA}/bin/python" \
    .. \
    && make -j 4 \
    && ln -s ${PREFIX_AUX}/spt3g/build/bin/* ${PREFIX_AUX}/bin/ \
    && ln -s ${PREFIX_AUX}/spt3g/build/spt3g ${PREFIX_AUX}/lib/python3.6/site-packages/ \
    && cd ${spt3g_start} \
    && rm -rf spt3g_software_sa

# Install TIDAS

# FIXME: commented out until TIDAS build bug fixed on ubuntu 16.04

 # git clone https://github.com/hpc4cmb/tidas.git --branch master --single-branch --depth 1 \
 #    && cd tidas \
 #    && mkdir build \
 #    && cd build \
 #    && cmake \
 #    -DCMAKE_C_COMPILER="mpicc" \
 #    -DCMAKE_CXX_COMPILER="mpicxx" \
 #    -DMPI_C_COMPILER="mpicc" \
 #    -DMPI_CXX_COMPILER="mpicxx" \
 #    -DCMAKE_C_FLAGS="-O3 -fPIC -pthread" \
 #    -DCMAKE_CXX_FLAGS="-O3 -fPIC -pthread" \
 #    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
 #    -DPYTHON_EXECUTABLE:FILEPATH="${PREFIX_CONDA}/bin/python" \
 #    -DCMAKE_INSTALL_PREFIX="${PREFIX_AUX}" \
 #    .. \
 #    && make -j 4 && make install \
 #    && cd ../.. \
 #    && rm -rf tidas

# Install PySM

 pip install https://github.com/bthorne93/PySM_public/archive/2.1.0.tar.gz --prefix=${PREFIX_AUX}

# Install PyMPIT for environment testing

 git clone https://github.com/tskisner/pympit.git \
    && cd pympit \
    && python setup.py build \
    && python setup.py install --prefix=${PREFIX_AUX} \
    && cd compiled \
    && CC=mpicc make \
    && cp pympit_compiled "${PREFIX_AUX}/bin/" \
    && cd ../.. \
    && rm -rf pympit

# Install PbArchive and dependencies

curl -SL https://www.libarchive.org/downloads/libarchive-3.3.2.tar.gz \
   | tar -xzf - \
   && cd libarchive-3.3.2 \
   && CC="gcc" CFLAGS="-O3 -fPIC -pthread" \
   ./configure --prefix="${PREFIX_AUX}" \
   && make -j 4 && make install \
   && cd .. \
   && rm -rf libarchive*

curl -SL http://portal.nersc.gov/project/cmb/kisner/pbarchive-0.5.8.tar.bz2 \
    | tar -xjf - \
    && cd pbarchive-0.5.8 \
    && PYTHON="python3" CC="gcc" CPPFLAGS="-I${PREFIX_AUX}/include" LDFLAGS="-L${PREFIX_AUX}/lib" \
    ./configure --prefix=${PREFIX_AUX} --with-libarchive=${PREFIX_AUX} \
    && make -j 4 \
    && make install \
    && cd .. \
    && rm -rf pbarchive*


# Compile python modules

python3 -m compileall -f "${PREFIX_CONDA}/lib/python3.6/site-packages"
python3 -m compileall -f "${PREFIX_AUX}"

# Install script to load this into the environment

setup="${PREFIX}/deps/${VERSION}/setup.sh"
echo "# Loads version ${VERSION} of toast deps into your environment" > ${setup}
echo "export TOAST_AUX_ROOT=${PREFIX_AUX}" >> ${setup}
echo "export CMAKE_PREFIX_PATH=${PREFIX_AUX}:${CMAKE_PREFIX_PATH}" >> ${setup}
echo "export PATH=${PREFIX_AUX}/bin:${PREFIX_CONDA}/bin:${PATH}" >> ${setup}
echo "export CPATH=${PREFIX_AUX}/include:${CPATH}" >> ${setup}
echo "export LIBRARY_PATH=${PREFIX_AUX}/lib:${LIBRARY_PATH}" >> ${setup}
echo "export LD_LIBRARY_PATH=${PREFIX_AUX}/lib:${LD_LIBRARY_PATH}" >> ${setup}
echo "export PYTHONPATH=${PREFIX_AUX}/lib/python3.6/site-packages:${PYTHONPATH}" >> ${setup}
echo "" >> ${setup}

# Set permissions

chgrp -R polarbear "${PREFIX}/deps/${VERSION}"
chmod -R g+rwX,a+rX "${PREFIX}/deps/${VERSION}"
