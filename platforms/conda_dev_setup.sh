#!/bin/bash

# If you would like to create your environments in an alternate
# directory, first load the base environment and add the
# alternate location to your conda config:
#
#    conda config --append envs_dirs path/to/envs
#
# Then run this script with the full location:
#
#    ./platforms/conda_dev_setup.sh path/to/envs/my_env
#
# and then when activating the environment you can reference
# it with just "my_env".
#
# NOTE:  by default, the full path to the conda env will be
# displayed in the shell prompt.  To avoid this, edit ~/.condarc
# and add this line:
#
# env_prompt: '({name}) '
#
# Now only the basename of the environment will show up.

envname=$1

# Install optional dependencies if desired
optional=$2
if [ "x${optional}" = "xyes" ]; then
    echo "Optional dependencies set to 'yes'"
else
    echo "Optional dependencies set to 'no' or unspecified"
fi

# Explicit python version to use
pyversion=$3
if [ "x${pyversion}" = "x" ]; then
    pyversion=3.10
fi

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1

conda_exe=$(which conda)
if [ "x${conda_exe}" = "x" ]; then
    echo "No conda executable found"
    exit 1
fi

usage () {
    echo ""
    echo "Usage:  $0 <name of conda env or path>"
    echo ""
    echo "The named environment will be activated (and created)"
    echo ""
}

if [ "x${envname}" = "x" ]; then
    usage
    exit 1
fi

# Activate the base environment
basedir=$(dirname $(dirname "${conda_exe}"))
source ${basedir}/etc/profile.d/conda.sh
conda deactivate
conda activate base

# Get the list of packages to install
pkgfile="${scriptdir}/conda_dev_pkgs.txt"
pkglist=$(cat "${pkgfile}" | xargs -I % echo -n '"%" ')
platform=$(python -c 'import sys; print(sys.platform)')
pkglist="python=${pyversion} ${pkglist} compilers conda-build libopenblas=*=*openmp* libblas=*=*openblas"

# Determine whether the environment is a name or a
# full path.
env_noslash=$(echo "${envname}" | sed -e 's/\///g')
is_path=no
if [ "${env_noslash}" != "${envname}" ]; then
    # This was a path
    is_path=yes
    env_check=""
    if [ -e "${envname}/bin/conda" ]; then
	# It already exists
	env_check="${envname}"
    fi
else
    env_check=$(conda env list | grep "${envname} ")
fi

# Packages to install with pip instead of conda
pip_list=""

if [ "x${env_check}" = "x" ]; then
    # Environment does not yet exist.  Create it.
    if [ ${is_path} = "no" ]; then
	echo "Creating new environment \"${envname}\""
	conda create --yes -n "${envname}"
    else
	echo "Creating new environment \"${envname}\""
	conda create --yes -p "${envname}"
    fi
    echo "Activating environment \"${envname}\""
    conda activate "${envname}"
    echo "Setting default channel in this env to conda-forge"
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
else
    echo "Activating environment \"${envname}\""
    conda activate "${envname}"
    conda env list
fi

echo "Installing conda packages:  ${pkglist}"
conda install --yes --update-all ${pkglist}
# The "cc" symlink breaks Crays...
rm -f "${CONDA_PREFIX}/bin/cc"

if [ "x${pip_list}" != "x" ]; then
    echo "Installing pip packages:  ${pip_list}"
    pip install ${pip_list}
fi

if [ "x${optional}" != "xyes" ]; then
    # we are done
    exit 0
fi

# Now we will install MPI support and some optional dependencies
# that depend on MPI compilers.

conda install --yes --update-all autoconf libtool automake mpi4py mpich libopenblas=*=*openmp* libblas=*=*openblas

# Install libmadam

version=1.0.2
pfile=libmadam-${version}.tar.bz2
curl -SL -o ${pfile} https://github.com/hpc4cmb/libmadam/releases/download/v${version}/${pfile}

if [ ! -f ${pfile} ]; then
    echo "Failed to fetch ${pfile}" >&2
    exit 1
fi

rm -rf libmadam-${version}
tar xjf ${pfile} \
    && pushd libmadam-${version} >/dev/null \
    && FC="mpif90" MPIFC="mpif90" FCFLAGS="-O3 -g -fPIC" \
    CC="mpicc" MPICC="mpicc" CFLAGS="-O3 -g -fPIC" \
    LDFLAGS="-L${CONDA_PREFIX}/lib -lfmpich -lgfortran" \
    ./configure --with-cfitsio="${CONDA_PREFIX}" \
    --with-fftw="${CONDA_PREFIX}" \
    --with-blas="-L${CONDA_PREFIX}/lib -lopenblas" \
    --with-lapack="-L${CONDA_PREFIX}/lib -lopenblas" \
    --prefix="${CONDA_PREFIX}" \
    && make -j 2 \
    && make install \
    && pushd python >/dev/null \
    && python3 setup.py install --prefix="${CONDA_PREFIX}" --old-and-unmanageable \
    && popd >/dev/null \
    && popd >/dev/null

rm -rf libmadam-${version}*


# Install libconviqt

version=1.2.7
pfile=libconviqt-${version}.tar.gz
curl -SL -o ${pfile} https://github.com/hpc4cmb/libconviqt/archive/v${version}.tar.gz

if [ ! -f ${pfile} ]; then
    echo "Failed to fetch ${pfile}" >&2
    exit 1
fi

rm -rf libconviqt-${version}
tar xzf ${pfile} \
    && pushd libconviqt-${version} >/dev/null \
    && ./autogen.sh \
    && CC="mpicc" CXX="mpicxx" MPICC="mpicc" MPICXX="mpicxx" \
    CFLAGS="-O3 -g -fPIC -std=gnu99" CXXFLAGS="-O3 -g -fPIC" \
    OPENMP_CFLAGS="-fopenmp" OPENMP_CXXFLAGS="-fopenmp" \
    LDFLAGS="-L${CONDA_PREFIX}/lib" \
    ./configure \
    --with-cfitsio="${CONDA_PREFIX}" --prefix="${CONDA_PREFIX}" \
    && make -j 2 \
    && make install \
    && pushd python >/dev/null \
    && python3 setup.py install --prefix="${CONDA_PREFIX}" --old-and-unmanageable \
    && popd >/dev/null \
    && popd >/dev/null

rm -rf libconviqt-${version}*
