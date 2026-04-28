#!/bin/bash

nbdir=$1
if [ "x${nbdir}" = "x" ]; then
    echo "Usage:  ${0} <docs dir>"
    exit 1
fi

# This script might be called collectively from an mpirun / srun.
# Since find is not guaranteed to produce results in a consistent
# order, we sort the results

all_nb=$(find "${nbdir}" -name "*.ipynb" | grep -v checkpoint | sort)

for nbfile in ${all_nb}; do
    echo "Cleaning ${nbfile}"
    nbstripout "${nbfile}"
    echo "Running ${nbfile} in place..."
    jupyter nbconvert --to notebook --execute "${nbfile}" --inplace
done
