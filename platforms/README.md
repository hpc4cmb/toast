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

## Using NVHPC for GPU Support with OpenMP Target

When using OpenMP target offload, the internal compiled extension is built with
a compiler (such as the NVHPC SDK compilers) that support both the modern
OpenMP standard and the GPU device on the system. TOAST also requires a BLAS /
LAPACK library, and this must be compatible with the OpenMP implementation used
to build TOAST. For this reason, we cannot simply link to the conda-provided
OpenBLAS (built with the conda compilers). Instead we must setup a software
stack which compiles our dependencies using the same custom compiler we are
using for the TOAST build.

We must also be careful to separate libraries installed by conda from libraries
built with our custom compilers. This allows us to keep the conda libraries out
of our library search path. On your system, first activate the conda base
environment and then load the NVHPC tools. The following example is on the
perlmutter system at NERSC:

    $> module load python
    $> module use /global/common/software/cmb/perlmutter/nvhpc/modulefiles
    $> module load nvhpc-nompi/23.7

Next, setup a build directory (for example, on scratch) and decide where to
install the software. For the purposes of this example, we will build software
in `${SCRATCH}/build_nvhpc` and install software to `${HOME}/toast_nvhpc`.

    $> cd ${SCRATCH}/build_nvhpc
    $> MPICC=cc MPICXX=CC MPIFC=ftn \
        ${HOME}/git/toast/platforms/conda_dev_setup_nvhpc.sh \
        ${HOME}/toast_nvhpc | tee log 2>&1

This will create a conda environment in `${HOME}/toast_nvhpc` and will also
create an "extra" directory in `${HOME}/toast_nvhpc_ext` which has the
additional compiled dependencies built with our custom compilers. To load both
the conda environment and this extra prefix into our environment, we can use a
small shell helper function defined in this source file here:

    $> source ${HOME}/git/toast/packaging/conda/load_conda_external.sh
    $> load_conda_ext ${HOME}/toast_nvhpc

Now we are finally ready to compile toast itself. This script will run cmake
and configure the installation to the "extra" directory for our conda
environment created above:

    $> cd ${SCRATCH}/build_nvhpc
    $> ${HOME}/git/toast/platforms/conda_dev_nvhpc.sh
    $> make -j 4 install

As always, run the unit tests to verify the installation. You can enable GPU
use by setting `TOAST_GPU_OPENMP=1` and `OMP_TARGET_OFFLOAD=MANDATORY` in your
shell environment.

## Enabling GPU Support with JAX

For any software stack (using conda compilers, NVHPC, etc), you can [install
JAX using the official instructions
here](https://github.com/google/jax#installation). You can enable JAX code by
setting `TOAST_GPU_JAX=1` in your shell environment.

