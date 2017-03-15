#!/bin/bash

infile=$1
outfile=$2
rules=$3
precommand=$4

rm -f "${outfile}"

while IFS='' read -r line || [[ -n "${line}" ]]; do
    match="no"
    for rule in ${rules}; do
        rulefile="rules/${rule}.sh"
        if [[ "${line}" =~ @${rule}@ ]]; then
            match="yes"
            echo -n "${precommand} " >> "${outfile}"
            cat "${rulefile}" >> "${outfile}"
            break
        fi
    done
    if [ "${match}" = "no" ]; then
        echo "${line}" >> "${outfile}"
    fi
done < "${infile}"


