#!/bin/sh

LIBTOAST_PATH="${1}"

sed \
-e "s#@LIBTOAST_PATH@#${LIBTOAST_PATH}#g" \
ctoast.py.in > ctoast.py

