#!/bin/bash

set -o errexit

# get the absolute path to the directory with this script
topdir=$(readlink -f $(realpath $(dirname ${BASH_SOURCE[0]})))
echo "Running in top directory: ${topdir}..."

if ! eval command -v cmake &> /dev/null ; then
    module load cmake
fi

LOADED_VTUNE="$(module list &> /dev/stdout | grep vtune | awk '{print $NF}')"
if [ -z "${LOADED_VTUNE}" ]; then
    echo "Please load the preferred vtune module"
    exit
fi

: ${account:=mp107}
: ${queue:=debug}
: ${vtune:=${LOADED_VTUNE}}

for type in satellite ground ground_simple; do
    for size in tiny small medium large; do
        for machine in cori-intel-knl cori-intel-haswell edison-intel; do
	
            template="${topdir}/templates/vtune_${type}.in"
            sizefile="${topdir}/templates/params/${type}.${size}"
            machfile="${topdir}/templates/machines/machine.${machine}"
            outfile="${topdir}/vtune_${size}_${type}_${machine}.slurm"

            echo "Generating ${outfile}"

            cmake -DINFILE=${template} -DOUTFILE=${outfile} \
                -DMACHFILE=${machfile} -DSIZEFILE=${sizefile} \
                -DTOPDIR="${topdir}" -DACCOUNT=${account} -DTYPE=${type} \
                -DSIZE=${size} -DMACHINE=${machine} -DQUEUE=${queue} \
                -DVTUNE="${vtune}" $@ \
                -P ${topdir}/templates/config.cmake
        done
    done
done






