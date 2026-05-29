#!/bin/bash

nbdir=$1
if [ "x${nbdir}" = "x" ]; then
    echo "Usage:  ${0} <docs dir>"
    exit 1
fi

export OMP_NUM_THREADS=1

launch=$2
if [ -z "${launch}" ]; then
    mpiexec=$(which mpirun)
    if [ -z "${mpiexec}" ]; then
        # No mpi, run serially
        launch=""
    else
        launch="${mpiexec} -np 1"
    fi
fi

all_nb=$(find "${nbdir}" -name "*.ipynb" | grep -v checkpoint | sort)

for nbfile in ${all_nb}; do
    echo "Cleaning ${nbfile}"
    nbstripout "${nbfile}"
    echo "Running ${nbfile} in place..."
    eval ${launch} jupyter nbconvert --to notebook --execute "${nbfile}" --inplace
done
