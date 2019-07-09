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
echo "Building dependencies with:"
echo "  CC = ${CC} $(${CC} -dumpversion)"
echo "  CXX = ${CXX} $(${CXX} -dumpversion)"
echo "  FC = ${FC} $(${FC} -dumpversion)"

# Install travis python
# if [ ! -e "${HOME}/virtualenv/python3.6/bin/activate" ]; then
#     wget https://s3.amazonaws.com/travis-python-archives/binaries/ubuntu/14.04/x86_64/python-3.6.tar.bz2
#     sudo tar xjf python-3.6.tar.bz2 --directory /
# fi
source ${HOME}/virtualenv/python3.6/bin/activate


# Clone the cmbenv package
git clone https://github.com/hpc4cmb/cmbenv.git
cd cmbenv
./cmbenv -c travis_14.04 -p ${HOME}/software

mkdir -p build
cd build
../install_travis_14.04.sh | tee log




# Package up tarball
cd "${HOME}" \
    && export \
    TARBALL="travis_14.04_gcc${toolchain}_python${PYSITE}.tar.bz2" \
    && tar cjf "${TARBALL}" software
