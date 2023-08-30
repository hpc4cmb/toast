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

echo "scripts in ${scriptdir}"
echo "deps in ${depsdir}"

if [ "x${CONDA_EXE}" = "x" ]; then
    export CONDA_EXE=$(which conda)
fi
if [ "x${CONDA_EXE}" = "x" ]; then
    echo "No conda executable found"
    exit 1
fi

echo "${CONDA_EXE}"

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

# Install conda packages

pkgfiles="${scriptdir}/deps.txt ${scriptdir}/extdeps.txt"
if [ "x${optional}" = "xyes" ]; then
    pkgfiles="${pkgfiles} ${scriptdir}/optdeps.txt"
fi

pkglist=""
for pfile in ${pkgfiles}; do
    plist=$(cat "${pfile}" | xargs -I % echo -n '"%" ')
    pkglist="${pkglist} ${plist}"
done
pkglist="python=${pyversion} ${pkglist} compilers conda-build"
echo "Installing conda packages:  ${pkglist}"
conda install --yes --update-all ${pkglist}
# The "cc" symlink breaks Crays...
rm -f "${CONDA_PREFIX}/bin/cc"

if [ "x${optional}" != "xyes" ]; then
    # we are done
    exit 0
fi

# Reload the environment to pick up compiler environment variables
conda deactivate
conda activate "${ENVNAME}"

# Install MPI, needed by some optional dependencies
. "${scriptdir}/install_mpi.sh"

# Now build the packages not available through conda

# CC set by conda compilers
# CXX set by conda compilers
# FC set by conda compilers
if [ "x${MPICC}" = "x" ]; then
    # The user did not specify MPI compilers, which means that
    # the conda mpich package was installed
    export MPICC=mpicc
    export MPICXX=mpicxx
    export MPIFC=mpif90
    export MPFCLIBS="-L${CONDA_PREFIX}/lib -lfmpich -lgfortran"
fi
export CFLAGS="-O3 -g -fPIC"
export CXXFLAGS="-O3 -g -fPIC"
export FCFLAGS="-O3 -g -fPIC"

platform=$(python -c 'import sys; print(sys.platform)')
if [ ${platform} = "linux" ]; then
    export OMPFLAGS="-fopenmp"
    export SHLIBEXT="so"
else
    export OMPFLAGS=""
    export SHLIBEXT="dylib"
fi

export DEPSDIR="${depsdir}"
export PREFIX="${CONDA_PREFIX}"
export MAKEJ=2
export STATIC=no
export CLEANUP=no

for pkg in "libmadam" "libconviqt"; do
    . "${depsdir}/${pkg}.sh"
done

