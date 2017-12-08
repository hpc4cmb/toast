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
    #if eval command -v module &> /dev/null ; then
        module load cmake
    #fi
fi

: ${account:=mp107}
: ${queue:=debug}
: ${TYPES:="satellite ground ground_simple"}
: ${SIZES:="tiny small medium large representative"}
: ${MACHINES:="cori-knl cori-haswell edison"}

for type in ${TYPES}; do
    for size in ${SIZES}; do
        for machine in ${MACHINES}; do    
            template="${topdir}/templates/${type}.in"
            sizefile="${topdir}/templates/params/${type}.${size}"
            machfile="${topdir}/templates/machines/${machine}"
            outfile="${topdir}/${size}_${type}_${machine}.slurm"

            echo "Generating ${outfile}"

            cmake -DINFILE=${template} -DOUTFILE=${outfile} \
                -DMACHFILE=${machfile} -DSIZEFILE=${sizefile} \
                -DTOPDIR="${topdir}" -DACCOUNT=${account} -DTYPE=${type} \
                -DSIZE=${size} -DMACHINE=${machine} -DQUEUE=${queue} $@ \
                -P ${topdir}/templates/config.cmake
        done
    done
done
