#!/bin/bash -l

#SBATCH --partition=regular
#SBATCH --constraint=knl,quad,cache
#SBATCH --core-spec=4
#SBATCH --account=mp107
#SBATCH --nodes=200
#SBATCH --time=00:30:00
#SBATCH --job-name=small-satellite
#SBATCH --output=cori-intel-knl_%j.log

echo Starting slurm script at $(date)

# This script assumes that you are running at NERSC and have already
# loaded the toast module for the correct machine / configuration.

# This should be the same as the --nodes option above
nodes=200

# How many processes are we running per node?
node_proc=32

# Generate the focalplane file if it does not already exist.

pstr="cori-intel-knl"
ndet="271"

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

groupsize=${node_proc}

outdir="out_${pstr}"

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

#--- Hardware configuration (no need to change) ----

# Hyperthread CPUs per physical core
cpu_per_core=4

# Physical cores we are using
node_cores=64

node_thread=$(( node_cores / node_proc ))
node_depth=$(( cpu_per_core * node_thread ))
procs=$(( nodes * node_proc ))

export OMP_NUM_THREADS=${node_thread}
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Set TMPDIR to be on the ramdisk
export TMPDIR=/dev/shm

run="srun --cpu_bind=cores -n ${procs} -N ${nodes} -c ${node_depth}"

echo Calling srun at $(date)

echo "${run} ${com}"
eval ${run} ${com} > "${outdir}.log" 2>&1

echo End slurm script at $(date)
