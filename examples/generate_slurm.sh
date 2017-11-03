#!/bin/bash

# get the absolute path to the directory with this script
pushd $(dirname $0) > /dev/null
topdir=$(pwd -P)
popd > /dev/null

: ${TYPES:="satellite ground ground_simple"}
: ${SIZES:="tiny small medium large"}
: ${MACHINES:="cori-intel-knl cori-intel-haswell edison-intel"}

for type in ${TYPES}; do
    for size in ${SIZES}; do
        for machine in ${MACHINES}; do
            template="${type}_template"
            sizefile="${type}.${size}"
            machfile="machine.${machine}"
            outfile="${size}_${type}_${machine}.slurm"

            echo "Generating ${outfile}"

            # Create list of substitutions

            confsub=""

            while IFS='' read -r line || [[ -n "${line}" ]]; do
                # is this line commented?
                comment=$(echo "${line}" | cut -c 1)
                if [ "${comment}" != "#" ]; then

                    check=$(echo "${line}" | sed -e "s#.*=.*#=#")
                
                    if [ "x${check}" = "x=" ]; then
                        # get the variable and its value
                        var=$(echo ${line} | sed -e "s#\([^=]*\)=.*#\1#" | awk '{print $1}')
                        val=$(echo ${line} | sed -e "s#[^=]*= *\(.*\)#\1#")
                        # add to list of substitutions
                        confsub="${confsub} -e 's#@@${var}@@#${val}#g'"
                    fi
                fi

            done < "${sizefile}"

            while IFS='' read -r line || [[ -n "${line}" ]]; do
                # is this line commented?
                comment=$(echo "${line}" | cut -c 1)
                if [ "${comment}" != "#" ]; then

                    check=$(echo "${line}" | sed -e "s#.*=.*#=#")
                
                    if [ "x${check}" = "x=" ]; then
                        # get the variable and its value
                        var=$(echo ${line} | sed -e "s#\([^=]*\)=.*#\1#" | awk '{print $1}')
                        val=$(echo ${line} | sed -e "s#[^=]*= *\(.*\)#\1#")
                        # add to list of substitutions
                        confsub="${confsub} -e 's#@@${var}@@#${val}#g'"
                    fi
                fi

            done < "${machfile}"

            # We add these predefined matches at the end- so that the config
            # files can also use these.

            confsub="${confsub} -e 's#@@machine@@#${machine}#g'"
            confsub="${confsub} -e 's#@@size@@#${size}#g'"
            confsub="${confsub} -e 's#@@topdir@@#${topdir}#g'"

            #echo "${confsub}"

            # Process the template

            rm -f "${outfile}"

            while IFS='' read -r line || [[ -n "${line}" ]]; do
                echo "${line}" | eval sed ${confsub} >> "${outfile}"
            done < "${template}"

        done
    done
done






