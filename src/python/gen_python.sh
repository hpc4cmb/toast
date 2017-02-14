#!/bin/sh

IN="${1}"
OUT="${2}"
VERSION="${3}"
LIBTOAST_PATH="${4}"

sed \
-e "s#@VERSION@#${VERSION}#g" \
-e "s#@LIBTOAST_PATH@#${LIBTOAST_PATH}#g" \
"${IN}" > "${OUT}"

