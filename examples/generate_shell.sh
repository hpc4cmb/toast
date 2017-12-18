#!/bin/bash

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

: ${TYPES:="satellite ground ground_simple ground_multisite"}
# don't allow customization
SIZES="tiny"

for TYPE in ${TYPES}; do
    for SIZE in ${SIZES}; do
    
        TEMPLATE="${TOPDIR}/templates/${TYPE}_shell.in"
        SIZEFILE="${TOPDIR}/templates/params/${TYPE}.${SIZE}"
        OUTFILE="${TOPDIR}/${SIZE}_${TYPE}_shell.sh"

        echo "Generating ${OUTFILE}"

        cmake -DINFILE=${TEMPLATE} -DOUTFILE=${OUTFILE} \
            -DSIZEFILE=${SIZEFILE} \
            -DTOPDIR="${TOPDIR}" -DTYPE=${TYPE} \
            -DSIZE=${SIZE} $@ \
            -P ${TOPDIR}/templates/config.cmake
            
        chmod 755 ${OUTFILE}
    done
done

