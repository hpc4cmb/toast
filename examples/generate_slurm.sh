#!/bin/bash -l

set -o errexit

# if no realpath command, then add function
if ! eval command -v realpath &> /dev/null ; then
    realpath()
    {
        [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
    }
fi 

# get the absolute path to the directory with this script
TOPDIR=$(realpath $(dirname ${BASH_SOURCE[0]}))

echo "Running in top directory: ${TOPDIR}..."

if ! eval command -v cmake &> /dev/null ; then
    echo -e "\n\tCommand: \"cmake\" not available. Unable to proceed...\n"
    exit 2
fi

: ${ACCOUNT:=mp107}
: ${QUEUE:=debug}
: ${TYPES:="satellite ground ground_simple ground_multisite"}
: ${SIZES:="tiny small medium large representative"}
: ${MACHINES:="cori-knl cori-haswell"}

for TYPE in ${TYPES}; do
    for SIZE in ${SIZES}; do
        for MACHINE in ${MACHINES}; do   
         
            TEMPLATE="${TOPDIR}/templates/${TYPE}.in"
            SIZEFILE="${TOPDIR}/templates/params/${TYPE}.${SIZE}"
            MACHFILE="${TOPDIR}/templates/machines/${MACHINE}"
            OUTFILE="${TOPDIR}/${SIZE}_${TYPE}_${MACHINE}.slurm"

            echo "Generating ${OUTFILE}"

            cmake -DINFILE=${TEMPLATE} -DOUTFILE=${OUTFILE} \
                -DMACHFILE=${MACHFILE} -DSIZEFILE=${SIZEFILE} \
                -DTOPDIR="${TOPDIR}" -DACCOUNT=${ACCOUNT} -DTYPE=${TYPE} \
                -DSIZE=${SIZE} -DMACHINE=${MACHINE} -DQUEUE=${QUEUE} $@ \
                -P ${TOPDIR}/templates/config.cmake
        done
    done
done

# set default to 0
: ${GENERATE_SHELL:=0}
# generate shell if set to greater than zero
if [ "${GENERATE_SHELL}" -gt 0 ]; then
    ${TOPDIR}/generate_shell.sh
fi

# set default to 0
: ${GENERATE_VTUNE:=0}
# generate VTune if set to greater than zero
if [ "${GENERATE_VTUNE}" -gt 0 ]; then
    ${TOPDIR}/generate_vtune.sh
fi
