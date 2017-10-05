#!/bin/bash

echo Starting script at $(date)

# This script assumes that you already have toast and all dependencies
# loaded into your environment.

# Generate the focalplane file if it does not already exist.

pstr="generic"

ndet="37"

fpfile="fp_${pstr}_${ndet}.pkl"
if [ ! -e "${fpfile}" ]; then
    srun -n 1 toast_fake_focalplane.py --minpix ${ndet} --out "fp_${pstr}"
fi

# The executable script

ex=$(which toast_satellite_sim.py)
echo "Using ${ex}"

# Scan strategy parameters from a file

parfile="../../params/satellite/sim_noise_hwp.par"

# Observations

obs_len="24.0"
obs_gap="4.0"
nobs="157"

# Map making parameters

nside="512"

# Data distribution parameters

outdir="out_${pstr}"

# The commandline

com="${ex} \
--fp ${fpfile} \
--nside ${nside} \
--obs ${obs_len} \
--gap ${obs_gap} \
--numobs ${nobs} \
--outdir out_${pstr} \
"

# How many MPI processes to use?  Here we use 4 mpi processes, each with
# 2 threads.  This should be fine for a quad-core machine with
# hyperthreading

procs=4
export OMP_NUM_THREADS=2

run="mpirun -np ${procs}"

echo Calling mpirun at $(date)

echo "${run} ${com}"
eval ${run} ${com} > "${outdir}.log" 2>&1

echo End script at $(date)

