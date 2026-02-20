# Installation

In order to use TOAST and develop external custom classes, you can install
binary packages from conda-forge or PyPI. If you are working on the TOAST
package itself, there are several options for setting up a development
environment.

## Binary Packages

If you are working in an arbitrary python environment and want to install a
self-contained version of the software, you can install TOAST from PyPI with:

    pip install toast

!!! note

    Prior to the 3.0 release, you will need to add the `--pre` option to the
    install command.

If you are using a python stack with the conda package manager, you can install
toast as a package from conda-forge. It is strongly recommended that you get
your base conda environment from conda-forge, which will automatically use the
conda-forge channel. Create a new environment for your work and then install
TOAST with:

    conda install toast

!!! note

    Prior to the 3.0 release, the toast package on conda-forge is out of date.
    Do not use.

## Building from Source

The easiest, cross-platform way of building TOAST from source and developing the
codebase is to use a conda environment to provide all dependencies. There are
helper scripts to accomplish this. On Linux (and likely MacOS with homebrew),
you can install dependencies from OS packages.

### Conda

The first step is to install the conda-forge "base" environment. [Instructions
are here](https://conda-forge.org/download/). You should never install
additional packages to the conda base environment. Always create one or more
environments for different purposes. This allows you to update / delete
individual environments as needed while not contaminating the base environment
where the conda tools themselves live.

If you already have a (possibly old) conda base environment, make sure it is up
to date with:

    conda update --all -n base

Make sure to activate the base environment before running the scripts below:

    conda activate base

Next, go into a checkout of the toast git repository. Decide what you want to
call the new conda environment we will create. For this example, we will call it
`toast-dev`. Use the included helper script to set up a new conda environment
and install toast dependendencies:

    cd toast
    ./platforms/conda_dev_setup.sh toast-dev 3.13 no

!!! note

    You can also replace `toast-dev` with the full path to the environment
    location. The python version is optional. If you specify "yes" for the final
    argument, libmadam and libconviqt will also be installed.

!!! note

    The script above will install the default conda MPI package. If you have a
    custom MPI compiler already, set the `MPICC` environment variable to that
    compiler before running the script.

After some waiting, the new conda environment will be set up.  Now activate it:

    conda activate toast-dev

Now we are ready to install the toast package. Currently this uses CMake, but
soon will be driven by `scikit-build-core`, so these instructions will change
slightly. From the top of the toast source tree, make a build directory:

    mkdir -p build_conda
    cd build_conda

Now use the helper script that runs CMake and gets dependencies from your
currently activate conda environment:

    ../platforms/conda_dev.sh

Next compile and install toast into your current conda environment:

    make -j 4 install

As you hack on the code, re-run that make command. Whenever you want to work on
toast, just activate the `toast-dev` conda environment and continue.

### OS Packages

!!! warning

    To-Do: discuss list of OS packages to install, creation of a virtualenv
    using the system python3, etc.
