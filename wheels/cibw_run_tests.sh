# This file is sourced by cibuildwheel using /bin/sh

# echo "Skipping tests to produce wheel artifacts for local testing"

echo "======================================================================="
echo "Running serial tests with MPI disabled"
echo "======================================================================="
MPI_DISABLE=1 python -c 'import toast.tests; toast.tests.run()'

# echo "======================================================================="
# echo "Running MPI tests with $(which mpirun)"
# echo "*** FIXME:  currently MPI run inside the cibuildwheel docker containers"
# echo "*** spawns MPI processes independently (they each have a comm world of"
# echo "*** one process.  So we are not really testing much.  Change this to"
# echo "*** two processes once it works."
# echo "======================================================================="
# com="mpirun -np 1 python -m mpi4py -c 'import toast.tests; toast.tests.run()'"
# echo ${com}
# eval ${com}
