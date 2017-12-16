#!/bin/bash

# get the absolute path to the directory with this script
pushd $(dirname $0) > /dev/null
topdir=$(pwd -P)
popd > /dev/null

for type in satellite ground ground_simple ground_multisite; do
    for size in tiny; do
        template="${type}_shell_template"
        sizefile="${type}.${size}"
        outfile="${size}_${type}_shell.sh"

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

        # We add these predefined matches at the end- so that the config
        # files can also use these.

        confsub="${confsub} -e 's#@@size@@#${size}#g'"
        confsub="${confsub} -e 's#@@topdir@@#${topdir}#g'"

        #echo "${confsub}"

        # Process the template

        rm -f "${outfile}"

        while IFS='' read -r line || [[ -n "${line}" ]]; do
            echo "${line}" | eval sed ${confsub} >> "${outfile}"
        done < "${template}"

        chmod +x "${outfile}"

    done
done
