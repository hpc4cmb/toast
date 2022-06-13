#!/bin/bash

#set -e

echo "======================================================================="
echo "Upgrade numpy / scipy to latest versions for runtime tests"
echo "======================================================================="
pip install --upgrade numpy scipy

echo "======================================================================="
echo "Running MPI tests with $(which mpirun)"
echo "*** FIXME:  currently MPI run inside the cibuildwheel docker containers"
echo "*** spawns MPI processes independently (they each have a comm world of"
echo "*** one process.  So we are not really testing much.  Change this to"
echo "*** two processes once it works."
echo "======================================================================="
com="mpirun -np 1 python -c 'import toast.tests; toast.tests.run()'"
echo ${com}
eval ${com}

echo "======================================================================="
echo "Running serial tests with MPI disabled"
echo "======================================================================="
com="MPI_DISABLE=1 python -c 'import toast.tests; toast.tests.run()'"
echo ${com}
eval ${com}
