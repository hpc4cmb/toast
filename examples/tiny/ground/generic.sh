#!/bin/bash

echo Starting script at $(date)

# This script assumes that you already have toast and all dependencies
# loaded into your environment.

# Generate the focalplane file if it does not already exist.

parfile="../../params/ground/focalplane_small.par"

pstr="generic"
fpfile="fp_${pstr}_19.pkl"

if [ ! -e "${fpfile}" ]; then
    python3 $(which toast_fake_focalplane.py) @$parfile --out "fp_${pstr}"
fi

# Generate the schedule file if it does not already exist.

parfile="../../params/ground/schedule_small.par"

pstr="generic"
schedulefile="schedule_${pstr}.txt"

if [ ! -e "${schedulefile}" ]; then
    python3 $(which toast_ground_schedule.py) @$parfile --out "${schedulefile}"
fi

# The executable script

ex=$(which toast_ground_sim.py)
echo "Using ${ex}"

# Scan strategy parameters from a file

parfile="../../params/ground/ground_sim_small.par"

# Data distribution parameters

outdir="out_${pstr}"
mkdir -p "${outdir}"

# The commandline

com="${ex} @${parfile} \
--fp ${fpfile} \
--schedule ${schedulefile} \
--out out_${pstr} \
"

# How many MPI processes to use?  Just one for this tiny case.
# The number of OpenMP threads should default to the number of
# hyperthreads on the system.

procs=1

run="mpirun -np ${procs}"

echo Calling mpirun at $(date)

echo "${run} ${com}"
eval ${run} ${com} > "${outdir}/log" 2>&1

echo End script at $(date)
