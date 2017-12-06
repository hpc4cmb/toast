#!/bin/bash

set -o errexit

# get the absolute path to the directory with this script
topdir=$(realpath $(dirname ${BASH_SOURCE[0]}))
echo "Running in top directory: ${topdir}..."

if ! eval command -v cmake &> /dev/null ; then
    module load cmake
fi

: ${TYPES:="satellite ground ground_simple"}
# don't allow customization
SIZES="tiny"

for type in ${TYPES}; do
    for size in ${SIZES}; do
        template="${topdir}/templates/${type}_shell.in"
        sizefile="${topdir}/templates/params/${type}.${size}"
        outfile="${topdir}/${size}_${type}_shell.sh"

        echo "Generating ${outfile}"

        cmake -DINFILE=${template} -DOUTFILE=${outfile} \
            -DSIZEFILE=${sizefile} \
            -DTOPDIR="${topdir}" -DTYPE=${type} \
            -DSIZE=${size} $@ \
            -P ${topdir}/templates/config.cmake
            
        chmod 755 ${outfile}
    done
done

