# Example Configurations

This directory contains a variety of examples for how to configure toast with
cmake using a variety of upstream dependencies. Below we cover the most common.

## Building with Conda Compilers

If you have a base conda environment loaded, you can create a new conda env
with all toast dependencies by doing:

    $> mkdir build
    $> cd build
    $> ../platforms/conda_dev_setup.sh <new env>

Where "new env" is the name or path of the environment to create and populate
with dependencies. Next, activate the new environment.

    $> conda activate <new env>

And now configure toast to use all the conda-provided dependencies and install
to the conda prefix:

    $> cd build
    $> ../platforms/conda_dev.sh
    $> make -j 4 install

## Using LLVM for GPU Support with OpenMP Target

When using OpenMP target offload, the internal compiled extension is built with
a compiler that support both the modern OpenMP standard and the GPU device on
the system. TOAST also requires a BLAS / LAPACK library, and this must be
compatible with the OpenMP implementation used to build TOAST.

For this example, we will use LLVM to build all our compiled dependencies. The
following example is on the perlmutter system at NERSC:

    $> module load python
    $> module use /global/common/software/cmb/perlmutter/nvhpc/modulefiles
    $> module load nvhpc-nompi/23.7

Next, setup a build directory (for example, on scratch) and decide where to
install the software. For the purposes of this example, we will build software
in `${SCRATCH}/build_nvhpc` and install software to `${HOME}/toast_nvhpc`.  We
are installing optional dependencies (MPI, libconviqt, etc) with the "yes"
option.

    $> cd ${SCRATCH}/build_nvhpc
    $> MPICC=cc MPICXX=CC MPIFC=ftn \
        ${HOME}/git/toast/platforms/venv_dev_setup_nvhpc.sh \
        ${HOME}/toast_nvhpc yes | tee log 2>&1

This will create a virtualenv in `${HOME}/toast_nvhpc` and then install our
compiled dependencies into that directory. To load these tools into our
environment, we can use a small shell helper function defined in this source
file here:

    $> source ${HOME}/git/toast/packaging/venv/load_venv.sh
    $> load_venv ${HOME}/toast_nvhpc

Now we are finally ready to compile toast itself. This script will run cmake
and configure the installation to the dependencies in our virtualenv:

    $> cd ${SCRATCH}/build_nvhpc
    $> ${HOME}/git/toast/platforms/venv_dev_nvhpc.sh
    $> make -j 4 install

As always, run the unit tests to verify the installation. You can enable GPU
use by setting `TOAST_GPU_OPENMP=1` and `OMP_TARGET_OFFLOAD=MANDATORY` in your
shell environment.

## Using NVHPC for GPU Support with OpenMP Target

When using OpenMP target offload, the internal compiled extension is built with
a compiler (such as the NVHPC SDK compilers) that support both the modern
OpenMP standard and the GPU device on the system. TOAST also requires a BLAS /
LAPACK library, and this must be compatible with the OpenMP implementation used
to build TOAST.

For this example, we will use the NVHPC-provided BLAS and LAPACK libraries and
compile other dependencies with the NVHPC compilers. The following example is
on the perlmutter system at NERSC:

    $> module load python
    $> module use /global/common/software/cmb/perlmutter/nvhpc/modulefiles
    $> module load nvhpc-nompi/23.7

Next, setup a build directory (for example, on scratch) and decide where to
install the software. For the purposes of this example, we will build software
in `${SCRATCH}/build_nvhpc` and install software to `${HOME}/toast_nvhpc`.  We
are installing optional dependencies (MPI, libconviqt, etc) with the "yes"
option.

    $> cd ${SCRATCH}/build_nvhpc
    $> MPICC=cc MPICXX=CC MPIFC=ftn \
        ${HOME}/git/toast/platforms/venv_dev_setup_nvhpc.sh \
        ${HOME}/toast_nvhpc yes | tee log 2>&1

This will create a virtualenv in `${HOME}/toast_nvhpc` and then install our
compiled dependencies into that directory. To load these tools into our
environment, we can use a small shell helper function defined in this source
file here:

    $> source ${HOME}/git/toast/packaging/venv/load_venv.sh
    $> load_venv ${HOME}/toast_nvhpc

Now we are finally ready to compile toast itself. This script will run cmake
and configure the installation to the dependencies in our virtualenv:

    $> cd ${SCRATCH}/build_nvhpc
    $> ${HOME}/git/toast/platforms/venv_dev_nvhpc.sh
    $> make -j 4 install

As always, run the unit tests to verify the installation. You can enable GPU
use by setting `TOAST_GPU_OPENMP=1` and `OMP_TARGET_OFFLOAD=MANDATORY` in your
shell environment.

## Enabling GPU Support with JAX

For any software stack (using conda compilers, NVHPC, etc), you can [install
JAX using the official instructions
here](https://github.com/google/jax#installation). You can enable JAX code by
setting `TOAST_GPU_JAX=1` in your shell environment.

