# Example Configurations

This directory contains a variety of examples for how to configure toast with
cmake using a variety of upstream dependencies. Below we cover the most common
use cases.

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
following example is on the perlmutter system at NERSC.  First load the LLVM
environment:

    $> module swap PrgEnv-gnu PrgEnv-llvm
    $> module swap llvm llvm/17
    # Load conda base just to get python and venv module
    $> module load python

Next, setup a build directory (for example, on scratch) and decide where to
install the software. For the purposes of this example, we will build software
in `${SCRATCH}/build_llvm` and install software to `${HOME}/toast_llvm`.

    $> cd ${SCRATCH}/build_llvm
    $> MPICC=cc \
        ${HOME}/git/toast/platforms/perlmutter-llvm_setup.sh \
        ${HOME}/toast_llvm | tee log 2>&1

This will create a virtualenv in `${HOME}/toast_llvm` and then install our
compiled dependencies into that directory.  Next we activate this virtualenv
and add it to our shared library search path:

    $> source ${HOME}/toast_llvm/bin/activate
    $> export LD_LIBRARY_PATH=${HOME}/toast_llvm/lib:${LD_LIBRARY_PATH}

Now we are finally ready to compile toast itself. This script will run cmake
and configure the installation to the dependencies in our virtualenv:

    $> cd ${SCRATCH}/build_llvm
    $> ${HOME}/git/toast/platforms/perlmutter-llvm.sh
    $> make -j 4 install

As always, run the unit tests to verify the installation. You can enable GPU
use by setting `TOAST_GPU_OPENMP=1` and `OMP_TARGET_OFFLOAD=MANDATORY` in your
shell environment.

### Running a Simple Benchmark

This example is run on a compute node (i.e. get an interactive node with
salloc).  First we try the ground-based benchmark:

    # Create the ground benchmark inputs
    OMP_NUM_THREADS=4 srun -n 1 toast_benchmark_ground_setup --work_dir inputs

    # Run the "tiny" case on the CPU
    OMP_NUM_THREADS=4 OMP_TARGET_OFFLOAD=DISABLED srun -n 4 \
        toast_benchmark_ground \
        --input_dir inputs \
        --out_dir out_tiny_cpu \
        --case tiny | tee log_tiny_cpu 2>&1

    # Run the "tiny" case on the GPU
    OMP_NUM_THREADS=4 OMP_TARGET_OFFLOAD=MANDATORY TOAST_GPU_OPENMP=1 srun -n 4 \
        toast_benchmark_ground \
        --input_dir inputs \
        --out_dir out_tiny_gpu \
        --case tiny | tee log_tiny_gpu 2>&1

We can also run the satellite case:

    # Run the "tiny" case on the CPU.
    # NOTE:  the first run will create some input files,
    # which takes some additional time.
    OMP_NUM_THREADS=4 OMP_TARGET_OFFLOAD=DISABLED srun -n 4 \
        toast_benchmark_satellite \
        --out_dir out_tiny_cpu \
        --case tiny | tee log_tiny_cpu 2>&1

    # Run the "tiny" case on the GPU
    OMP_NUM_THREADS=4 OMP_TARGET_OFFLOAD=MANDATORY TOAST_GPU_OPENMP=1 srun -n 4 \
        toast_benchmark_satellite \
        --out_dir out_tiny_gpu \
        --case tiny | tee log_tiny_gpu 2>&1

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
    $> module load nvhpc-nompi/23.9

Next, setup a build directory (for example, on scratch) and decide where to
install the software. For the purposes of this example, we will build software
in `${SCRATCH}/build_nvhpc` and install software to `${HOME}/toast_nvhpc`.

    $> cd ${SCRATCH}/build_nvhpc
    $> MPICC=cc \
        ${HOME}/git/toast/platforms/perlmutter-nvhpc_setup.sh \
        ${HOME}/toast_nvhpc | tee log 2>&1

This will create a virtualenv in `${HOME}/toast_nvhpc` and then install our
compiled dependencies into that directory. Next we activate this virtualenv
and add it to our shared library search path:

    $> source ${HOME}/toast_nvhpc/bin/activate
    $> export LD_LIBRARY_PATH=${HOME}/toast_nvhpc/lib:${LD_LIBRARY_PATH}

Now we are finally ready to compile toast itself. This script will run cmake
and configure the installation to the dependencies in our virtualenv:

    $> cd ${SCRATCH}/build_nvhpc
    $> ${HOME}/git/toast/platforms/perlmutter-nvhpc.sh
    $> make -j 4 install

As always, run the unit tests to verify the installation. You can enable GPU
use by setting `TOAST_GPU_OPENMP=1` and `OMP_TARGET_OFFLOAD=MANDATORY` in your
shell environment.

## Enabling GPU Support with JAX

For any software stack (using conda compilers, NVHPC, etc), you can [install
JAX using the official instructions
here](https://github.com/google/jax#installation). You can enable JAX code by
setting `TOAST_GPU_JAX=1` in your shell environment.

