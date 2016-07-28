.. _nersc:

Using at NERSC
====================

To use TOAST at NERSC, you need to have a Python3 software stack with all dependencies installed.  There is already such a software stack installed on edison and cori.

Module Files
---------------

To get access to the needed module files, add the machine-specific module file location to your search path::

    %> module use /global/common/${NERSC_HOST}/contrib/hpcosmo/modulefiles

Load Dependencies
--------------------

In order to load a full python-3.5 stack, and also all dependencies needed by toast, do::

    %> module load toast-deps

