#!/bin/bash

set -o errexit

calc()
{
    awk "BEGIN{ print $* }"
}

convert_time()
{
    local STR=$(echo $1 | sed 's/:/ /g' | sed 's/=/ /g' | sed 's/"//g')
    local HOUR=$(echo $STR | awk '{print $2}' | sed 's/^00/0/g')
    local MIN=$(echo $STR | awk '{print $3}' | sed 's/^00/0/g')
    local TIME=$(calc "(${HOUR}*60 + ${MIN})")
    echo "${TIME}"
}

# default to not output
: ${INCLUDE_OUTPUT:=1}

: ${SRUN_ARGS=""}
: ${ARGS:="$@"}
: ${SLURM_SALLOC_COMMAND:=@SLURM_SALLOC_COMMAND@}

if [ "${SLURM_SALLOC_COMMAND}" = "SLURM_SALLOC_COMMAND-NOTFOUND" ]
then
    SLURM_SALLOC_COMMAND=salloc
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
    else
        chmod a+x ${i}
    fi
done
#------------------------------------------------------------------------------#
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

echo -e "\nRunning \"${ARGS}\"...\n"

eval ${ARGS}
