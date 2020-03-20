#!/bin/bash
#
# This is the script that travis actually runs.  We put these commands in a
# separate script (rather than in .travis.yml) so that we can make sure to
# echo things to travis to keep it running.
#

# Abort on Error
set -e

# # Time between keep-alive statements
# export KEEPALIVE=60s
#
# # Run the loop
# bash -c "while true; do echo \$(date) - Running tests...; sleep $KEEPALIVE; done" &
# KEEPALIVE_PID=$!
#
# error_handler() {
#     echo "ERROR: An error was encountered while running tests."
#     # We could do other cleanup / dumping here...
#     kill $KEEPALIVE_PID
#     exit 1
# }
#
# # If an error occurs, run our error handler
# trap 'error_handler' ERR

# Run MPI tests
#===========================================================================

export OMP_NUM_THREADS=2

# Run tests with MPI
#===========================================================================

mpirun -np 2 python3 -c "import toast.tests; toast.tests.run()"

# Run tiny MPI example pipelines
#===========================================================================

cd examples/
# Disable tests with PySM until https://github.com/astropy/astropy/pull/9190
# is merged or otherwise resolved
#bash run_tiny_tests.sh
TYPES="ground ground_simple" bash run_tiny_tests.sh
cd ..

#===========================================================================
# End tests

# # Cleanly terminate the keep alive loop
# kill $KEEPALIVE_PID
