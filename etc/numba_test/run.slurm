#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --job-name=test-numba
#SBATCH --licenses=SCRATCH
#SBATCH --constraint=knl
#SBATCH --core-spec=4
#SBATCH --account=mp107

ulimit -c unlimited

export OMP_NUM_THREADS=8
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

let nnode=2
let ntask_node=64/$OMP_NUM_THREADS
#let ntask_node=32/$OMP_NUM_THREADS
let ntask=$nnode*$ntask_node
let ncore=4*$OMP_NUM_THREADS

#export NUMBA_NUM_THREADS=$OMP_NUM_THREADS

com="srun -n $ntask -c $ncore --cpu_bind=cores numba_threads.py"
echo ${com}
eval ${com}
