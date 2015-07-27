# pytoast
Python Time Ordered Astrophysics Scalable Tools

## Quick Start

Make sure you have dependencies installed:  python3, mpi4py, numpy, 
setuptools, ...  Then run

    $> python setup.py develop --prefix=/somewhere/in/python/path

Now you can run the test suite (note the "quiet" option to disable annoying
from every process):

    $> mpirun -np 4 python setup.py -q test

