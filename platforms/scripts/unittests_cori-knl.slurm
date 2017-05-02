#!/bin/bash -l

#SBATCH --partition=debug
#SBATCH --constraint=knl,quad,cache
#SBATCH --account=mp107
#SBATCH --nodes=1
#SBATCH --core-spec=4
#SBATCH --time=00:30:00
#SBATCH --job-name=toastunit

# THIS MUST BE THE SAME AS --nodes ABOVE
NODES=1

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

com="python -c \"import toast; toast.test()\""

echo "${run} ${com}"
eval "${run} ${com}"

