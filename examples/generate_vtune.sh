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
topdir=$(realpath $(dirname ${BASH_SOURCE[0]}))

echo "Running in top directory: ${topdir}..."

if ! eval command -v cmake &> /dev/null ; then
    module load cmake
fi

LOADED_VTUNE="$(module list &> /dev/stdout | grep vtune | awk '{print $NF}')"
NO_SLURM=0
if [ -z "${LOADED_VTUNE}" ]; then
    echo "Only generating shell scripts. To Generate SLURM scripts at NERSC, please load the preferred vtune module"
    NO_SLURM=1
fi

: ${account:=mp107}
: ${queue:=debug}
: ${vtune:=${LOADED_VTUNE}}
: ${TYPES:="satellite ground ground_simple"}
: ${SIZES:="tiny small medium large representative"}
: ${MACHINES:="cori-knl cori-haswell edison"}

# if "tiny" is not in SIZES then disable "tiny" in SHELL_SIZES
SHELL_SIZES="tiny"
if [ -z "$(echo ${SIZES} | grep tiny)" ]; then
    SHELL_SIZES=""
fi

# Generate Shell
for type in ${TYPES}; do
    for size in ${SHELL_SIZES}; do

        if [ -z "${size}" ]; then continue; fi
        
        template="${topdir}/templates/vtune_${type}_shell.in"
        sizefile="${topdir}/templates/params/${type}.tiny"
        outfile="${topdir}/vtune_${size}_${type}_shell.sh"

        echo "Generating ${outfile}"

        cmake -DINFILE=${template} -DOUTFILE=${outfile} \
            -DSIZEFILE=${sizefile} \
            -DTOPDIR="${topdir}" -DTYPE=${type} \
            -DSIZE=${size} \
            -DVTUNE="${vtune}" $@ \
            -P ${topdir}/templates/config.cmake
        
    done
done

# skip the SLURM generation
if [ "${NO_SLURM}" -eq 1 ]; then
    exit 0
fi

# generate SLURM
for type in ${TYPES}; do
    for size in ${SIZES}; do
        for machine in ${MACHINES}; do
	
            template="${topdir}/templates/vtune_${type}.in"
            sizefile="${topdir}/templates/params/${type}.${size}"
            machfile="${topdir}/templates/machines/${machine}"
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
