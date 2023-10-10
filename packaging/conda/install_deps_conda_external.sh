#!/bin/bash

set -e

envname=$1

# Explicit python version to use
pyversion=$2
if [ "x${pyversion}" = "x" ]; then
    pyversion=3.10
fi

# Install optional dependencies if desired
optional=$3
if [ "x${optional}" = "xyes" ]; then
    echo "Optional dependencies set to 'yes'"
else
    echo "Optional dependencies set to 'no' or unspecified"
fi

# Location of this script and dependencies
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
depsdir=$(dirname ${scriptdir})/deps

if [ "x${CONDA_EXE}" = "x" ]; then
    export CONDA_EXE=$(which conda)
fi
if [ "x${CONDA_EXE}" = "x" ]; then
    echo "No conda executable found"
    exit 1
fi

usage () {
    echo ""
    echo "Usage:  $0 <name of conda env or path> <python version> <extra deps (yes/no)>"
    echo ""
    echo "The named environment will be created or activated"
    echo ""
}

if [ "x${envname}" = "x" ]; then
    usage
    exit 1
fi

export ENVNAME=${envname}

# Create / activate env
. "${scriptdir}/create_env.sh"

# Install conda packages.  We only install the things needed from python.

pkgfiles="${scriptdir}/deps.txt"
pkglist=""
for pfile in ${pkgfiles}; do
    plist=$(cat "${pfile}" | xargs -I % echo -n '"%" ')
    pkglist="${pkglist} ${plist}"
done
pkglist="python=${pyversion} ${pkglist}"
echo "Installing conda packages:  ${pkglist}"
conda install --yes --update-all ${pkglist}

# Reload the environment to pick up any environment variables
conda deactivate
conda activate "${ENVNAME}"

# Add our compiled prefix into our search environment

prepend_env () {
    # This function is needed since trailing colons
    # on some environment variables can cause major
    # problems...
    local envname="$1"
    local envval="$2"
    eval "local temp=\"\${$envname}\""
    if [ -z ${temp+x} ]; then
        export ${envname}="${envval}"
    else
        export ${envname}="${envval}:${temp}"
    fi
}

export PREFIX="${CONDA_PREFIX}_ext"
mkdir -p "${PREFIX}/bin"
mkdir -p "${PREFIX}/include"
mkdir -p "${PREFIX}/lib"
prepend_env "PATH" "${PREFIX}/bin"
prepend_env "CPATH" "${PREFIX}/include"
prepend_env "LIBRARY_PATH" "${PREFIX}/lib"
prepend_env "LD_LIBRARY_PATH" "${PREFIX}/lib"
if [ ! -e "${PREFIX}/lib64" ]; then
    ln -s "${PREFIX}/lib" "${PREFIX}/lib64"
fi

# Compile dependencies with variables optionally set in the calling environment

if [ "x${CC}" = "x" ]; then
    export CC=gcc
fi
if [ "x${CXX}" = "x" ]; then
    export CXX=g++
fi
if [ "x${FC}" = "x" ]; then
    export FC=gfortran
fi

if [ "x${CFLAGS}" = "x" ]; then
    export CFLAGS="-O3 -g -fPIC"
fi
if [ "x${CXXFLAGS}" = "x" ]; then
    export CXXFLAGS="-O3 -g -fPIC"
fi
if [ "x${FCFLAGS}" = "x" ]; then
    export FCFLAGS="-O3 -g -fPIC"
fi

platform=$(python -c 'import sys; print(sys.platform)')
if [ ${platform} = "linux" ]; then
    if [ "x${OMPFLAGS}" = "x" ]; then
        export OMPFLAGS="-fopenmp"
    fi
    export SHLIBEXT="so"
else
    export SHLIBEXT="dylib"
fi

if [ "x${MAKEJ}" = "x" ]; then
    export MAKEJ=2
fi

export DEPSDIR="${depsdir}"
export STATIC=no
export CLEANUP=no

if [ "x${LAPACK_LIBRARIES}" = "x" ]; then
    export BLAS_LIBRARIES="-L${PREFIX}/lib -lopenblas ${OMPFLAGS} -lm ${FCLIBS}"
    export LAPACK_LIBRARIES="-L${PREFIX}/lib -lopenblas ${OMPFLAGS} -lm ${FCLIBS}"
    . "${depsdir}/openblas.sh"
fi

for pkg in cfitsio fftw libflac suitesparse libaatm; do
    . "${depsdir}/${pkg}.sh"
done

if [ "x${MPICC}" = "x" ]; then
    # The user did not specify MPI compilers- try to use generic
    # defaults.
    echo "====================================================================="
    echo ""
    echo "WARNING:  If using custom compilers, you should have an"
    echo "MPI installation with C, C++, and Fortran compiler wrappers."
    echo "Set the MPICC, MPICXX, MPIFC, and MPIFCLIBS environment"
    echo "variables before using this script.  Assuming GNU compilers"
    echo "and MPICH for now..."
    export MPICC=mpicc
    export MPICXX=mpicxx
    export MPIFC=mpif90
    export MPFCLIBS="-L${CONDA_PREFIX}/lib -lfmpich -lgfortran"
    echo ""
    echo "====================================================================="
fi

# Install mpi4py, needed by some optional dependencies
. "${depsdir}/mpi4py.sh"

if [ "x${optional}" != "xyes" ]; then
    # we are done
    exit 0
fi

for pkg in "libmadam" "libconviqt"; do
    . "${depsdir}/${pkg}.sh"
done

echo "====================================================================="
echo ""
echo "Additional compiled dependencies installed next to conda"
echo "environment at '${PREFIX}'.  You should prepend the following"
echo "locations to your environment before compiling toast:"
echo ""
echo "   PATH --> '${PREFIX}/bin'"
echo "   CPATH --> '${PREFIX}/include'"
echo "   LIBRARY_PATH --> '${PREFIX}/lib'"
echo "   LD_LIBRARY_PATH --> '${PREFIX}/lib'"
echo "   PYTHONPATH --> '${PREFIX}/lib/python${pyversion}/site-packages'"
echo ""
echo "You can use the helper function in this directory:"
echo ""
echo "  %> source packaging/conda/load_conda_external.sh"
echo "  %> load_conda_ext ${ENVNAME}"
echo ""
echo "====================================================================="

