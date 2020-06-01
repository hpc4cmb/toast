#!/bin/bash

set -e

echo "Starting shell script at $(date)"

# Numba threading may conflict with our own.  Disable it.
export NUMBA_NUM_THREADS=1

# Pre-downloaded cache of PySM data files
export PYSM_LOCAL_DATA=$(pwd)/data/pysm-data

echo "Dumping current shell and TOAST environment to env.log ..."
echo -e "\n----------------- Current Shell Environment -------------------\n" > env.log
env | sort -d >> env.log
echo -e "---------------------------------------------------------------\n" >> env.log
echo "Python executable: $(which python3)" >> env.log
echo "Python version: $(python3 --version &> /dev/stdout)" >> env.log
echo "" >> env.log

# nodes used by this job
NODES=@nodes@

# set procs and threads
NODE_PROC=@node_procs@
PROC_THREADS=@omp_threads@
PROC_DEPTH=$(( @node_slots@ / NODE_PROC ))

# total number of processes on all nodes
NPROC=$(( NODES * NODE_PROC ))

echo "Using ${NODES} node(s), which have @node_slots@ thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

export OMP_NUM_THREADS=${PROC_THREADS}

# The launching command and options
launch_str="@mpi_launch@"
if [ "x@mpi_procs@" != "x" ]; then
    launch_str="${launch_str} @mpi_procs@ ${NPROC}"
fi
if [ "x@mpi_nodes@" != "x" ]; then
    launch_str="${launch_str} @mpi_nodes@ ${NODES}"
fi
if [ "x@mpi_depth@" != "x" ]; then
    launch_str="${launch_str} @mpi_depth@ ${PROC_DEPTH}"
fi
launch_str="${launch_str} @mpi_extra@"

# Trigger astropy downloads
#echo "Triggering astropy downloads..."
#python3 trigger_astropy.py

# Generate the Sky model, if needed

@PYSM_SKY@

# Log the TOAST runtime environment launched identically to the pipeline
echo "TOAST Runtime Environment:" >> env.log
ex=$(which toast_env_test.py)
com="${launch_str} ${ex} --groupsize @group_size@"
echo ${com} >> env.log
eval ${com} 2>&1 >> env.log

# Generate the telescope properties (focalplanes and observing schedules)

@TELESCOPES@

# Run the pipeline script

ex=$(which @pipe_script@)
echo "Using pipeline ${ex}"

com="${launch_str} ${ex} @pipeline.par --group-size @group_size@ @focalplane_list@ @schedule_list@"

echo ${com}

echo "Launching pipeline at $(date)"
eval ${com} > log 2>&1

echo "Ending batch script at $(date)"
