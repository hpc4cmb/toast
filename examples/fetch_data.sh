#!/bin/bash

RSYNC="rsync -a"
WGET="wget -nv -d -r -c -N -np -nH --cut-dirs=3 --reject \"index.html*\""

LOCAL_DATA="/project/projectdirs/cmb/www/toast_data"
REMOTE_DATA="http://portal.nersc.gov/project/cmb/toast_data"

# Get the absolute path to the directory with this script
pushd $(dirname $0) > /dev/null
base=$(pwd -P)
popd > /dev/null

# Output data directory
DATA="${base}/data"
mkdir -p "${DATA}"
pushd "${DATA}" > /dev/null

# Are we running at NERSC?
if [ -d "${LOCAL_DATA}" ]; then
    # yes
    eval ${RSYNC} "${LOCAL_DATA}/*" "."
else
    # no
    eval ${WGET} "${REMOTE_DATA}"
    rm robots.txt toast_data
fi

popd > /dev/null

