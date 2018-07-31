#!/bin/bash
#
# This config assumes use of clang and Anaconda python,
# using mpich, mpi4py, cfitsio, and fftw from the
# conda-forge channel.
#
# 1.  Install miniconda / anaconda
#
# 2.  Edit ~/.condarc to look like this:
#
#     channels:
#         - conda-forge
#         - defaults
#
# 3.  Install numpy, mpich, mpi4py, cfitsio, fftw
#
# 4.  In order to link against the relocatable shared
#     libraries installed by conda packages (which have
#     relative @rpaths), you must add the conda
#     environment lib directory to $DYLD_LIBRARY_PATH.
#     This is needed (for example) to use FFTW and CFITSIO
#     libraries installed by conda packages.
#

OPTS="$@"

condabin=$(dirname $(which python))

export PYTHON=python3
export CC=mpicc
export CXX=mpicxx
export MPICC=mpicc
export MPICXX=mpicxx
export CFLAGS="-O3 -g"
export CXXFLAGS="-O3 -g"

./configure ${OPTS} \
--without-mkl
