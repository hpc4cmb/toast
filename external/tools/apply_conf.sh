#!/bin/bash

infile=$1
outfile=$2
conffile=$3
prefix=$4
version=$5
moddir=$6
docker=$7

compiled_prefix="${prefix}/toast-deps/${version}_aux"
conda_prefix="${prefix}/toast-deps/${version}_conda"
module_dir="${moddir}/toast-deps"

if [ "x${docker}" = "xyes" ]; then
    compiled_prefix="/usr"
    conda_prefix="/usr"
fi

# Create list of substitutions

confsub="-e 's#@CONFFILE@#${conffile}#g'"

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
            confsub="${confsub} -e 's#@${var}@#${val}#g'"
        fi
    fi

done < "${conffile}"

# We add these predefined matches at the end- so that the config
# file can actually use these.

confsub="${confsub} -e 's#@AUX_PREFIX@#${compiled_prefix}#g'"
confsub="${confsub} -e 's#@CONDA_PREFIX@#${conda_prefix}#g'"
confsub="${confsub} -e 's#@VERSION@#${version}#g'"
confsub="${confsub} -e 's#@MODULE_DIR@#${module_dir}#g'"

rm -f "${outfile}"

while IFS='' read -r line || [[ -n "${line}" ]]; do
    echo "${line}" | eval sed ${confsub} >> "${outfile}"
done < "${infile}"
