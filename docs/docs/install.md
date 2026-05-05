# Installation

In order to use TOAST and develop external custom classes, you can install binary
packages from conda-forge or PyPI.  If you are working on the TOAST package itself,
there are several options for setting up a development environment.

## Binary Packages

If you are working in an arbitrary python environment and want to install a
self-contained version of the software, you can install TOAST from PyPI with:

    pip install toast

If you are using a python stack with the conda package manager, you can install toast as
a package from conda-forge.  It is strongly recommended that you get your base conda
environment from conda-forge, which will automatically use the conda-forge channel.
Create a new environment for your work and then install TOAST with:

    conda install toast

## Building from Source

The easiest, cross-platform way of building TOAST from source and developing the
codebase is to use a conda environment to provide all dependencies.  There are helper
scripts to accomplish this.  On Linux (and likely MacOS with homebrew), you can
alternatively install dependencies from OS packages.

### Conda Dependencies

The first step is to install the conda-forge "base" environment.  [Instructions are
here](https://conda-forge.org/download/).  The "base" environment contains the conda
tools and their immediate dependencies.  You should never install additional packages to
the conda base environment.  Always create one or more environments for different
purposes.  This allows you to update / delete individual environments as needed while
not contaminating or breaking the base environment where the conda tools themselves
live.

If you already have a (possibly old) conda-forge base environment, make sure it is up to
date with:

    conda update --all -n base

Make sure to activate the base environment before running the scripts below:

    conda activate base

Next, go into a checkout of the toast git repository.  Decide what you want to call the
new conda environment we will create.  For this example, we will call it `toast-dev`.
Use the included helper script to set up a new conda environment and install toast
dependendencies:

    cd toast
    ./conda_setup.sh -e toast-dev -p 3.13

!!! note

    You can also replace a name like `toast-dev` with the full path to the environment
    location.  The python version is optional.  If you specify `-x` then "extra"
    dependencies (including libmadam and libconviqt) will also be installed.

!!! note

    The script above will install the default conda MPI package.  If you have a custom
    MPI compiler already, set the `MPICC` environment variable to that compiler before
    running the script.

Another example, creating a conda environment on a Cray system where the MPI compiler is
called `cc` and where we want to install libmadam (MPI fortran code) and libconviqt (MPI
C++ code):

    MPICC=cc MPICXX=CC MPIFC=ftn ./conda_setup.sh -e toast-dev -p 3.13 -x

After some waiting, the new conda environment will be set up.  Now activate it:

    conda activate toast-dev

Now we are ready to install the toast package.  You can install toast once with:

    pip install -v .

This will compile all C++ sources and build a python wheel and then install that wheel.
If you are actively hacking on the code, you can install in "development mode" which
just installs links to the build directory. In development mode, if you change compiled
source files, these files will be recompiled on the next import. To install in
development / editable mode:

    pip install -v -e .

If you need to uninstall this editable version (for example to test out a new stable
version from conda-forge, etc), just use pip to uninstall toast first.

!!! note

    If you are using a conda environment created from scratch without using the
    conda_setup.sh helper script (for example by conda installing toast to get
    dependencies), then you may need to install additional packages when installing
    in editable mode.  Conda install scikit-build-core, pybind11, cmake, and ninja
    before using `pip install -e`.

### OS Package Dependencies

It is not possible to document installation instructions for every flavor of UNIX-like
operating system. Here we outline the packages to install in a Debian / Ubuntu OS to
provide all dependencies needed. Start by installing all required compiled dependencies:

    sudo apt update
    sudo apt install \
        build-essential \
        cmake \
        libfftw3-dev \
        libopenblas-openmp-dev \
        python3 \
        python3-dev

When we create our python environment later, we will usually want to install `mpi4py`,
since most of the parallelism in TOAST makes extensive use of MPI. You should install an
MPI flavor with your OS package manager so that mpi4py can find it later. The optional
`libconviqt` and `libmadam` packages below will also need this. For example:

    sudo apt install \
        libmpich-dev \
        mpich

#### Optional:  Atmosphere Simulation

If you want to simulate timestreams of atmospheric pickup, you will need to install
several other compiled packages, including SuiteSparse and the AATM library. These need
to be installed *before* installing TOAST. SuiteSparse is available as a package:

    sudo apt install libsuitesparse-dev

The `libaatm` package [can be found here](https://github.com/hpc4cmb/libaatm). Follow
instructions to compile and install that into a location that can be found by your user
account when installing TOAST.

#### Create a Virtualenv and Install

Make sure that the system python3 has the venv module:

    sudo apt install python3-venv

Using the system python3, create a virtualenv for our work and activate it:

    python3 -m venv ~/toast-dev
    source ~/toast-dev/bin/activate

You almost certainly want to install `mpi4py` as well, although it is optional:

    python3 -m pip install mpi4py

Then install TOAST to that virtualenv

    cd toast
    pip install -v .

If you want to install TOAST in editable mode, make sure you have some build
dependencies installed persistently in your environment:

    pip install scikit-build-core pybind11 cmake ninja

Then editable mode should work:

    cd toast
    pip install -v -e .

#### Optional:  Beam Convolution with Conviqt

The `libconviqt` package enables simulating timestreams from a 4-PI arbitrary beam
convolution of the sky. That package [is available
here](https://github.com/hpc4cmb/libconviqt). Follow instructions to compile and install
the library and python interface into a location that can be found by your user account
at run time. We install this *after* setting up our virtualenv so that we can install
the python bindings into our virtualenv. Before installing `libconviqt` you should
install CFITSIO:

    sudo apt install libcfitsio-dev

#### Optional:  Mapmaking with MADAM

The MADAM destriping mapmaker was developed and used for Planck data analysis. TOAST
includes a wrapper that can call the python interface to `libmadam` if it is available.
You can [download the package here](https://github.com/hpc4cmb/libmadam). Libmadam
requires the CFITSIO package as well.

## Verifying the Installation

You should always run the test suite (especially if you have compiled from source), just
to verify that everything is working. Start by testing with one process and a couple
threads to not overload your machine:

    OMP_NUM_THREADS=2 python -c 'import toast.tests; toast.tests.run()'

Then you can try using more processes. We do not routinely run the unit tests on more
than 4 processes currently. Although they should work, some tests may not have
sufficient amounts of data to distribute over many processes:

    OMP_NUM_THREADS=2 mpirun -np 4 python -c 'import toast.tests; toast.tests.run()'

