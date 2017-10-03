#!/bin/bash

echo Starting script at $(date)

# This script assumes that you already have toast and all dependencies
# loaded into your environment.

# Generate the focalplane file if it does not already exist.

pstr="generic"
fpfile="fp_${pstr}_1.pkl"

if [ ! -e "${fpfile}" ]; then
    toast_fake_focalplane.py --minpix 1 --output "fp_${pstr}"
fi

# The executable script

ex=$(which toast_satellite_sim.py)
echo "Using ${ex}"

# Scan strategy parameters from a file

parfile="../../params/satellite/sim_noise_hwp.par"

# Observations

obs_len="24.0"
obs_gap="4.0"
nobs="4"

# Map making parameters

nside="512"

# Data distribution parameters

outdir="out_${pstr}"
mkdir -p "${outdir}"

# The commandline

com="${ex} \
--groupsize ${groupsize} \
--fp ${fpfile} \
--nside ${nside} \
--obs ${obs_len} \
--gap ${obs_gap} \
--numobs ${nobs} \
--outdir out_${pstr} \
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

