#!/bin/bash -l

#SBATCH --partition=debug
#SBATCH --constraint=knl,quad,cache
#SBATCH --account=mp107
#SBATCH --nodes=4
#SBATCH --core-spec=4
#SBATCH --time=00:30:00
#SBATCH --job-name=fakesat

# THIS MUST BE THE SAME AS --nodes ABOVE
NODES=4

# processes per node
NODE_PROC=32
NPROC=$(( NODES * NODE_PROC ))

# NOTE we use 4 cores per node for specialization.
NODE_CPU_PER_CORE=4
NODE_CORE=64
NODE_THREAD=$(( NODE_CORE / NODE_PROC ))
NODE_DEPTH=$(( NODE_CPU_PER_CORE * NODE_THREAD ))

export OMP_NUM_THREADS=${NODE_THREAD}

# treat each hyperthread as a place an OpenMP task can go
export OMP_PLACES=threads

# spread threads as widely as possible (avoid sharing cores,cache etc)
export OMP_PROC_BIND=spread

run="srun -n ${NPROC} -N ${NODES} -c ${NODE_DEPTH} --cpu_bind=cores"

# The focalplane pickle file should be generated with the
# toast_fake_focalplane.py script.

# Since we have about ~1200 detectors, we group the processes into sizes
# of 4 nodes.

groupsize=${NODE_PROC}

com="toast_satellite_sim.py \
--groupsize ${groupsize} \
--samplerate 23.0 \
--spinperiod 10.0 \
--spinangle 30.0 \
--precperiod 93.0 \
--precangle 65.0 \
--hwprpm 88.0 \
--obs 24.0 \
--gap 4.0 \
--numobs 4 \
--debug \
--outdir out_fake-satellite-madam \
--nside 512 \
--fp fp_fake.pkl \
--madam \
--baseline 60.0 \
"

echo "${run} ${com}"
eval "${run} ${com}"

