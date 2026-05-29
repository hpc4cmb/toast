# Building the Documentation

This directory contains documentation built with zensical (mkdocs replacement). The
jupyter notebooks in this git repo have been stripped so that the binary outputs are not
stored in version control. Building the docs involves running all the notebooks and then
either building the docs (for deployment) or serving the docs (for local development).

## Dependencies

Activate a conda environment and ensure that the toast version you are building docs for
is installed with all dependencies. Then additionally install the dependencies for the
documentation. For example:

    cd toast
    ./conda_setup.sh -e toast-dev -p 3.13
    conda activate toast-dev
    pip install -v .
    conda install --file docs/doc_requirements.txt

Or at NERSC:

    module load python
    cd toast
    MPICC=cc ./conda_setup.sh -e toast-dev -p 3.13
    conda activate toast-dev
    pip install -v .
    conda install --file docs/doc_requirements.txt

## Running Notebooks

This operation can be potentially expensive and will be usually be done on a system with
MPI in order to run some larger examples. From a workstation or interactive compute
node, run the notebooks with (for example):

    cd docs
    ./run_notebooks.sh ./docs

Or at NERSC:

    # Get an interactive session...
    salloc ...
    cd docs
    ./run_notebooks.sh ./docs 'srun -N 1 -n 16'

## Build and Deploy

Build the docs and parse the notebooks into the final output directory:

    cd docs
    ./build.sh

Now push to github:

    ./deploy.sh

## Development Mode

When working on the docs, it is useful to view the "live" changes as you make them.
First, run all notebooks using the script above. Then, if you are actively working on a
notebook, make sure you have executed that and saved it. Next, build the docs and serve
it:

    cd docs
    ./serve.sh

