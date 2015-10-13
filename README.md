# pytoast
Python Time Ordered Astrophysics Scalable Tools

## Quick Start on Workstation

Make sure you have dependencies installed:  python3, mpi4py, numpy, 
setuptools, healpy, quaternionarray  Then run

    $> python setup.py develop --prefix=/somewhere/in/python/path

Now you can run the test suite (note the "quiet" option to disable 
annoying messages from every process):

    $> mpirun -np 4 python setup.py -q test


## Quick Start on edison.nersc.gov

Here is how to use a python-3.4 software stack which has all the 
dependencies needed by pytoast.   Since
  setup.py is MPI-aware (due to the unit test runner), you will have to
  install the develop hooks from a compute node

* From a stock module setup (no use of HPCPorts), do:

    $> . /scratch1/scratchdirs/kisner/software/python-3.4_gcc-4.9_mkl-13/setup.sh

* Now checkout pytoast and decide where to install it.  For this
  example, I'll assume you install to ${SCRATCH}/software/dev. 
