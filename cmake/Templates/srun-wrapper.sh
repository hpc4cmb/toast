#!/bin/bash

set -o errexit

# default to not output
: ${INCLUDE_OUTPUT:=1}

: ${SRUN_ARGS=""}
: ${ARGS:="$@"}
: ${SLURM_SRUN_COMMAND:=@SLURM_SRUN_COMMAND@}

if [ "${SLURM_SRUN_COMMAND}" = "SLURM_SRUN_COMMAND-NOTFOUND" ]
then
    SLURM_SRUN_COMMAND=srun
fi

for i in $(echo $@ | egrep '\.slurm')
do
    if [ ! -f "${i}" ]; then 
        if [ "$(echo $i | grep '^-')" ]; then
            if [ "${i}" = "--include-output" ]; then 
                INCLUDE_OUTPUT=0
                ARGS="$(echo ${ARGS} | sed 's/--include-output//g')"
            elif [ "${i}" = "--exclude-output" ]; then 
                INCLUDE_OUTPUT=1 
                ARGS="$(echo ${ARGS} | sed 's/--exclude-output//g')"
            fi
        fi
        continue
    fi
    
    for j in $(grep '^#SBATCH ' ${i} | sed 's/#SBATCH//g' | sed 's/  / /g')
    do
        if [ "${INCLUDE_OUTPUT}" -gt 0 -a ! -z "$(echo $j | grep 'output=')" ]
        then
            continue
        fi
        SRUN_ARGS="${SRUN_ARGS} ${j}"
    done

    SRUN_ARGS="$(echo ${SRUN_ARGS} | sed 's/^ //g')"
done

# set environment paths
export LOG_OUT=/dev/stdout
export PATH="${PROJECT_BIN_PATH}:@CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_BINDIR@:@PATH@"
export PYTHONPATH="${PROJECT_PYC_PATH}:@CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_PYTHONDIR@:@PYTHONPATH@"

if [ "$(uname -s)" = "Darwin" ]; 
then
    export DYLD_LIBRARY_PATH="${PROJECT_LIB_PATH}:@CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_LIBDIR@:@DYLD_LIBRARY_PATH@"
else
    export LD_LIBRARY_PATH="${PROJECT_LIB_PATH}:@CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_LIBDIR@:@LD_LIBRARY_PATH@"
fi

${SLURM_SRUN_COMMAND} ${SRUN_ARGS} ${ARGS}
