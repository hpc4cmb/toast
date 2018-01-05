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
    FILE="${outdir}/CTestNotes.cmake"

    echo "Creating CTest notes file: \"${FILE}\"..."
    cat > ${FILE} << EOF

IF(NOT DEFINED CTEST_NOTES_FILES)
    SET(CTEST_NOTES_FILES )
ENDIF(NOT DEFINED CTEST_NOTES_FILES)

EOF

    for i in ${outdir}/timing_report*.out
    do
        cat >> ${FILE} << EOF
LIST(APPEND CTEST_NOTES_FILES "${i}")
EOF
    done
    # remove duplicates
    cat >> ${FILE} << EOF

IF(NOT "\${CTEST_NOTES_FILES}" STREQUAL "")
    LIST(REMOVE_DUPLICATES CTEST_NOTES_FILES)
ENDIF(NOT "\${CTEST_NOTES_FILES}" STREQUAL "")

EOF

done

set +o errexit
set +v

PLOTS_SCRIPT=$(dirname ${BASH_SOURCE[0]})/generate_plots.sh
if [ -x "${PLOTS_SCRIPT}" ]; then
    eval ${PLOTS_SCRIPT} $@
fi
