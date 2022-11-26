#!/bin/bash

# Number of processes to use
nproc=$1

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
dir=$(pwd)
popd >/dev/null 2>&1

# Python work script
pyscript="${dir}/test_mpi_exit_code_work.py"

# Run it
echo "======================================================================="
echo "Testing MPI Abort() with $(which mpirun) on ${nproc} processes"
echo "======================================================================="
com="mpirun -np ${nproc} python -m mpi4py ${pyscript}"
echo ${com}
eval ${com}

# Verify non-zero exit code
status=$?
if (( $status == 0 )); then
    echo "FAIL:  Shell exit code was zero"
    exit 1
else
    echo "Shell exit code was ${status}"
    exit 0
fi
