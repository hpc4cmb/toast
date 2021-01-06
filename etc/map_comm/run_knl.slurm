#!/bin/bash -l

#SBATCH --partition=debug
#SBATCH --constraint=knl,quad,cache
#SBATCH --account=mp107
#SBATCH --nodes=1
#SBATCH --core-spec=4
#SBATCH --time=00:30:00
#SBATCH --job-name=mapcomm

set -e

echo "Starting batch script at $(date)"

# Set TMPDIR to be on the ramdisk
export TMPDIR=/dev/shm

# Numba threading may conflict with our own.  Disable it.
export NUMBA_NUM_THREADS=1

# nodes used by this job
NODES=${SLURM_JOB_NUM_NODES}

# set procs and threads
NODE_PROC=16
PROC_THREADS=4
PROC_DEPTH=$(( 256 / NODE_PROC ))

# total number of processes on all nodes
NPROC=$(( NODES * NODE_PROC ))

echo "Using ${NODES} node(s), which have 256 thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

export OMP_NUM_THREADS=${PROC_THREADS}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# The launching command and options
launch_str="srun"
if [ "x-n" != "x" ]; then
    launch_str="${launch_str} -n ${NPROC}"
fi
if [ "x-N" != "x" ]; then
    launch_str="${launch_str} -N ${NODES}"
fi
if [ "x-c" != "x" ]; then
    launch_str="${launch_str} -c ${PROC_DEPTH}"
fi
launch_str="${launch_str} --cpu_bind=cores"

# Run the pipeline script

export TOAST_FUNCTIME=1

for nside in 512 1024 2048 4096; do
    com="${launch_str} python toast_map_comm.py --nside ${nside}"
    echo ${com}
    echo "Launching pipeline at $(date)"
    eval ${com} > log 2>&1
done

echo "Ending batch script at $(date)"
