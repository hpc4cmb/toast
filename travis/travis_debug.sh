#!/bin/bash
#
# This script is designed to be run *INSIDE* a docker container that has
# been launched like:
#
#    %> ./travis_docker.sh <instance ID>
#
# Once you are inside the docker container, you can then run this script with
#
#    %> ./scripts/travis_debug.sh <gcc suffix>
#
# Where "gcc suffix" is something like "-7", "-6", "-5", or "".
# This will download and extract the dependency tarball and create a
# shell snippet ("toast_env.sh") that can be sourced to load the software
# into your environment.
#
# This also does a git clone of toast and creates a file toast/config.sh
# which can be used to run cmake from a toast/build directory for installing
# toast into the ~/software directory above.
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

# Activate the virtualenv and install the dependency tarball.

source ${HOME}/virtualenv/python3.6/bin/activate

# Set python site version
export PYSITE=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
echo "Python site version = ${PYSITE}"

# Get dependencies from the tarball.

wget http://portal.nersc.gov/project/cmb/toast_travis/travis_14.04_gcc${toolchain}_python${PYSITE}.tar.bz2

tar xjf travis_14.04_gcc${toolchain}_python${PYSITE}.tar.bz2 --directory ${HOME}

# Clone the repo if needed

if [ ! -d "${HOME}/toast" ]; then
    git clone https://github.com/hpc4cmb/toast.git
fi

# Create a shell snippet that sets up the build environment

env="${HOME}/toast_env.sh"

echo "# Source me before building" > "${env}"
echo "" >> "${env}"
echo 'export PATH="${HOME}/software/cmbenv_python/bin:${PATH}"' >> "${env}"
echo 'source cmbenv' >> "${env}"
echo "" >> "${env}"
echo "export CC=\$(which gcc${toolchain})" >> "${env}"
echo "export CXX=\$(which g++${toolchain})" >> "${env}"
echo "export FC=\$(which gfortran${toolchain})" >> "${env}"
echo "" >> "${env}"
echo "export MPICC=\$(which mpicc)" >> "${env}"
echo "export MPICXX=\$(which mpicxx)" >> "${env}"
echo "export MPIFC=\$(which mpif90)" >> "${env}"
echo "" >> "${env}"

conf="${HOME}/toast/config.sh"

echo '#!/bin/bash' > "${conf}"
echo 'cmake \' >> "${conf}"
echo '-DCMAKE_C_COMPILER="${CC}" \' >> "${conf}"
echo '-DCMAKE_CXX_COMPILER="${CXX}" \' >> "${conf}"
echo '-DMPI_C_COMPILER="${MPICC}" \' >> "${conf}"
echo '-DMPI_CXX_COMPILER="${MPICXX}" \' >> "${conf}"
echo '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \' >> "${conf}"
echo '-DCMAKE_INSTALL_PREFIX=${HOME}/software/cmbenv_aux \' >> "${conf}"
echo '..' >> "${conf}"
chmod +x "${conf}"
