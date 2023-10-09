#!/bin/bash

set -e

envname=$1

# Install optional dependencies if desired
optional=$2
if [ "x${optional}" = "xyes" ]; then
    echo "Optional dependencies set to 'yes'"
else
    echo "Optional dependencies set to 'no' or unspecified"
fi

# Static linking?
static=$3
if [ "x${optional}" = "xyes" ]; then
    echo "Static linking set to 'yes'"
else
    echo "Static linking set to 'no' or unspecified"
fi

# Location of this script and dependencies
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
depsdir=$(dirname ${scriptdir})/deps

PY_EXE=$(which python3)
if [ "x${PY_EXE}" = "x" ]; then
    echo "No python3 executable found"
    exit 1
fi

usage () {
    echo ""
    echo "Usage:  $0 <path to virtualenv> <extra deps (yes/no)>"
    echo ""
    echo "The named environment will be created and / or activated"
    echo ""
}

if [ "x${envname}" = "x" ]; then
    usage
    exit 1
fi

if [ ! -d "${envname}" ]; then
    # Create it
    python3 -m venv "${envname}"
fi

if [ ! -e "${envname}/bin/activate" ]; then
    echo "Environment \"${envname}\" exists, but activate script not found"
    exit 1
fi

. "${envname}/bin/activate"
python3 -m pip install install --upgrade pip setuptools wheel

# Install packages

pkglist=$(cat "${scriptdir}/deps.txt" | xargs -I % echo -n '% ')
echo "Installing pip packages:  ${pkglist}"
python3 -m pip install --no-input ${pkglist}

# Add our compiled prefix into our search environment

export PREFIX="${envname}"

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
export STATIC=${static}
export CLEANUP=no

if [ "x${LAPACK_LIBRARIES}" = "x" ]; then
    export BLAS_LIBRARIES="-L${PREFIX}/lib -lopenblas ${OMPFLAGS} -lm ${FCLIBS}"
    export LAPACK_LIBRARIES="-L${PREFIX}/lib -lopenblas ${OMPFLAGS} -lm ${FCLIBS}"
    . "${depsdir}/openblas.sh"
fi

for pkg in cfitsio fftw libflac suitesparse libaatm; do
    . "${depsdir}/${pkg}.sh"
done

if [ "x${optional}" != "xyes" ]; then
    # we are done
    exit 0
fi

# Install mpi4py, needed by some optional dependencies
. "${depsdir}/mpi4py.sh"

# Now build the packages

if [ "x${MPICC}" = "x" ]; then
    # This should have already raised an error installing mpi4py...
    export MPICC=mpicc
fi
if [ "x${MPICC}" = "x" ]; then
    export MPICXX=mpicxx
fi
if [ "x${MPIFC}" = "x" ]; then
    export MPIFC=mpif90
fi
if [ "x${MPFCLIBS}" = "x" ]; then
    export MPFCLIBS="-lfmpich -lgfortran"
fi

for pkg in "libmadam" "libconviqt"; do
    . "${depsdir}/${pkg}.sh"
done

