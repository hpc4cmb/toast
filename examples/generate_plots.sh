#!/bin/bash

set -o errexit

# if no realpath command, then add function
if ! eval command -v realpath &> /dev/null ; then
    realpath()
    {
        [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
    }
fi

# if the glob is unsuccessful, don't pass ${outdir}/timing_report*.out
shopt -s nullglob
for j in $@
do
    outdir=$(realpath ${j})

    for i in ${outdir}/timing_report*.json
    do
        ${PWD}/timing_plot.py -f ${i}
    done

    for i in ${outdir}/timing_report*.png
    do
        # show in log
        fname="$(basename $(dirname ${i}))/$(basename ${i})"
        cat << EOF
<DartMeasurementFile name="${fname}"
type="image/png">${i}</DartMeasurementFile>
EOF
    done
done

