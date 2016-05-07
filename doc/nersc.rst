.. _nersc:

Using at NERSC
====================

To use TOAST at NERSC, you need to have a Python3 software stack with all dependencies installed.  There are several ways to achieve this.


Temporary Solution
--------------------

For now, you can use a python 3.4 software stack built by Ted on edison and cori.  On edison::

    %> module use /scratch1/scratchdirs/kisner/software/modulefiles
    %> module load toast

OR::

    %> module use /scratch1/scratchdirs/kisner/software/modulefiles
    %> module load toast-deps

And then build and install your own copy of toast.  On cori, the path is different::

    %> module use /global/cscratch1/sd/kisner/software/modulefiles
    %> module load toast

OR::

    %> module use /global/cscratch1/sd/kisner/software/modulefiles
    %> module load toast-deps


NERSC Intel Python / Anaconda
---------------------------------

We will be moving towards a solution using NERSC-supported python3 within a conda environment, with some extra packages.  TBD.

