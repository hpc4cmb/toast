#!/bin/sh

IN="${1}"
OUT="${2}"
VERSION="${3}"
LIBTOAST_PATH="${4}"
HAVE_MPI="${5}"

MPISTR="False"
if [ "x${HAVE_MPI}" = "x1" ]; then
    MPISTR="True"
fi

sed \
-e "s#@VERSION@#${VERSION}#g" \
-e "s#@LIBTOAST_PATH@#${LIBTOAST_PATH}#g" \
-e "s#@HAVE_MPI@#${MPISTR}#g" \
"${IN}" > "${OUT}"

