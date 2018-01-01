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

# determine if we should generate SLURM scripts
LOADED_VTUNE="$(module list &> /dev/stdout | grep vtune | awk '{print $NF}')"
NO_SLURM=0
if ! command -v sbatch &> /dev/null; then
    echo ""
    echo "WARNING! Only generating shell scripts. \"sbatch\" command does not exist"
    echo ""
    NO_SLURM=1
elif [ -z "${LOADED_VTUNE}" ]; then
    echo ""
    echo "WARNING! Only generating shell scripts. To Generate SLURM scripts at"\
    "NERSC, please load the preferred vtune module"
    echo ""
    NO_SLURM=1
fi

: ${ACCOUNT:=mp107}
: ${QUEUE:=debug}
: ${VTUNE:=${LOADED_VTUNE}}
: ${TYPES:="satellite ground ground_simple"}
: ${SIZES:="tiny small medium large representative"}
: ${MACHINES:="cori-knl cori-haswell edison"}

# if "tiny" is not in SIZES then disable "tiny" in SHELL_SIZES
SHELL_SIZES="tiny"
if [ -z "$(echo ${SIZES} | grep tiny)" ]; then
    SHELL_SIZES=""
fi

: ${GENERATE_SHELL:=1}
export GENERATE_SHELL

# Determine "mpirun" available for shell scripts
: ${MPI_RUN:=mpirun}
for i in mpirun mpiexec lamexec srun
do
    if command -v ${i} &> /dev/null; then
        MPI_RUN=$(which ${i})
        break
    fi
done

# Generate Shell
if [ "${GENERATE_SHELL}" -gt 0 ]; then
    for TYPE in ${TYPES}; do
        for SIZE in ${SHELL_SIZES}; do
    
            if [ -z "${SIZE}" ]; then continue; fi
            
            TEMPLATE="${TOPDIR}/templates/vtune_${TYPE}_shell.in"
            SIZEFILE="${TOPDIR}/templates/params/${TYPE}.tiny"
            OUTFILE="${TOPDIR}/vtune_${SIZE}_${TYPE}_shell.sh"
    
            echo "Generating ${OUTFILE}"
    
            cmake -DINFILE=${TEMPLATE} -DOUTFILE=${OUTFILE} \
                -DSIZEFILE=${SIZEFILE} -DTOPDIR="${TOPDIR}" -DTYPE=${TYPE} \
                -DSIZE=${SIZE} -DVTUNE="${VTUNE}" -DMPI_RUN=${MPI_RUN} $@ \
                -P ${TOPDIR}/templates/config.cmake
            
            chmod 755 ${OUTFILE}
        done
    done
fi

# skip the SLURM generation
if [ "${NO_SLURM}" -eq 1 ]; then
    exit 0
fi

# generate SLURM
for TYPE in ${TYPES}; do
    for SIZE in ${SIZES}; do
        for MACHINE in ${MACHINES}; do
	
            TEMPLATE="${TOPDIR}/templates/vtune_${TYPE}.in"
            SIZEFILE="${TOPDIR}/templates/params/${TYPE}.${SIZE}"
            MACHFILE="${TOPDIR}/templates/machines/${MACHINE}"
            OUTFILE="${TOPDIR}/vtune_${SIZE}_${TYPE}_${MACHINE}.slurm"

            echo "Generating ${OUTFILE}"

            cmake -DINFILE=${TEMPLATE} -DOUTFILE=${OUTFILE} \
                -DMACHFILE=${MACHFILE} -DSIZEFILE=${SIZEFILE} \
                -DTOPDIR="${TOPDIR}" -DACCOUNT=${ACCOUNT} -DTYPE=${TYPE} \
                -DSIZE=${SIZE} -DMACHINE=${MACHINE} -DQUEUE=${QUEUE} \
                -DVTUNE="${vtune}" $@ \
                -P ${TOPDIR}/templates/config.cmake
        done
    done
done
