#!/bin/bash
#
# Run this script to upload a dependency tarball to NERSC, for future download
# by travis tests.  Run with your NERSC user ID:
#
#    %> ./scripts/travis_upload.sh <NERSC user ID>
#
# You must be in the "cmb" filegroup for this to work.
#

usage () {
    echo "$0 <NERSC user ID>"
    exit 1
}

nid="$1"
if [ "x${nid}" = "x" ]; then
    usage
    exit 1
fi

remote="/project/projectdirs/cmb/www/toast_travis/"

rsync -a -v -e ssh travis_*gcc*python*.tar.bz2 "${nid}@dtn01.nersc.gov:${remote}"
