#!/bin/bash

prefix=$1
deps=$2
pyver=$3

# Script directory
pushd $(dirname $0) > /dev/null
topdir=$(pwd)
popd > /dev/null

# Git version
gitdesc=$(git describe --tags --dirty --always | cut -d "-" -f 1)
gitcnt=$(git rev-list --count HEAD)
version="${gitdesc}.dev${gitcnt}"

# Create list of substitutions

sub="-e 's#@VERSION@#${version}#g'"
sub="${sub} -e 's#@PYVERSION@#${pyver}#g'"
sub="${sub} -e 's#@PREFIX@#${prefix}#g'"
sub="${sub} -e 's#@DEPS@#${deps}#g'"

out="mod_toast-${version}"

while IFS='' read -r line || [[ -n "${line}" ]]; do
    echo "${line}" | eval sed ${sub} >> "${out}"
done < "${topdir}/modulefile_toast.in"

out="modver_toast-${version}"

while IFS='' read -r line || [[ -n "${line}" ]]; do
    echo "${line}" | eval sed ${sub} >> "${out}"
done < "${topdir}/version.in"

