#!/bin/bash

# Abort on Error
set -e

RSYNC="rsync -avrP"

WGET_DATA="wget -nv -r -c -N -np -nH --progress=bar --cut-dirs=4 --reject \"index.html*\""
NERSC_DATA="/global/cfs/cdirs/cmb/www/toast_data/example_data"
REMOTE_DATA="http://portal.nersc.gov/project/cmb/toast_data/example_data"

WGET_PYSM="wget -nv -r -c -N -np -nH --progress=bar --cut-dirs=4 --reject \"index.html*\" --reject \".git\""
NERSC_PYSM="/global/cfs/cdirs/cmb/www/pysm-data"
REMOTE_PYSM="http://portal.nersc.gov/project/cmb/pysm-data"

# We do not (yet) need to pre-cache healpy data, since we are not running PySM
# on the fly.  Re-enable this once we are.
#
# WGET_HEALPY="wget -nv -r -c -N -np -nH --progress=bar --cut-dirs=3 --reject \"index.html*\" --reject \".git\""
# NERSC_HEALPY="/global/cfs/cdirs/cmb/www/healpy-data"
# REMOTE_HEALPY="http://portal.nersc.gov/project/cmb/healpy-data"


# Get the absolute path to the directory with this script
pushd $(dirname $0) > /dev/null
base=$(pwd -P)
popd > /dev/null

# Output data directory
DATA="${base}/data"
if [ "x$1" != "x" ]; then
    DATA="$1"
fi
mkdir -p "${DATA}"
pushd "${DATA}" > /dev/null

# Are we running at NERSC?
if [ -d "${NERSC_DATA}" ]; then
    # yes, rsync local copy into place
    echo "Running at NERSC, rsyncing / symlinking data..."
    eval ${RSYNC} "${NERSC_DATA}/*" "." &> /dev/stdout
    # Symlink the PYSM data
    ln -s "${NERSC_PYSM}" "./pysm-data"
else
    # no, use recursive wget
    echo "Not running at NERSC, fetching example data with wget..."
    eval ${WGET_DATA} "${REMOTE_DATA}" &> /dev/stdout
    rm -f robots.txt
    mkdir -p "pysm-data"
    pushd "pysm-data"
    echo "Not running at NERSC, fetching PySM data with wget..."
    eval ${WGET_PYSM} "${REMOTE_PYSM}/pysm_2_test_data" &> /dev/stdout
    eval ${WGET_PYSM} "${REMOTE_PYSM}/pysm_2" &> /dev/stdout
    rm -f robots.txt
    popd > /dev/null
fi

popd > /dev/null
