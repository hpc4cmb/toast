#!/bin/bash

prefix=$1
if [ "x${prefix}" = "x" ]; then
    echo "Usage: $0 <top-level prefix> <suffix>"
    exit 0
fi

suffix=$2
if [ "x${suffix}" = "x" ]; then
    echo "WARNING: no suffix given, using empty string"
else
    suffix="-${suffix}"
fi

# Script directory
pushd $(dirname $0) > /dev/null
topdir=$(pwd)
popd > /dev/null

# Python version
pyver=$(python --version 2>&1 | awk '{ print $2 }' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
if [ "x${pyver}" = "x" ]; then
    echo "cannot find python version"
    exit 1
fi

# Git version
gitdesc=$(git describe --tags --dirty --always | cut -d "-" -f 1)
gitcnt=$(git rev-list --count HEAD)
version="${gitdesc}.dev${gitcnt}${suffix}"

# toast-deps version
deps=""
mods=$(echo $LOADEDMODULES | tr ":" " ")
for mod in ${mods}; do
    ver=$(echo ${mod} | sed -e "s#toast-deps/##")
    if [ ${mod} != ${ver} ]; then
	deps="${ver}"
    fi
done
if [ "x${deps}" = "x" ]; then
    echo "The toast-deps module is not loaded!"
    exit 1
fi

# Create list of substitutions

sub="-e 's#@VERSION@#${version}#g'"
sub="${sub} -e 's#@PYVERSION@#${pyver}#g'"
sub="${sub} -e 's#@PREFIX@#${prefix}#g'"
sub="${sub} -e 's#@DEPS@#${deps}#g'"

out="mod_toast-${version}"
rm -f ${out}

while IFS='' read -r line || [[ -n "${line}" ]]; do
    echo "${line}" | eval sed ${sub} >> "${out}"
done < "${topdir}/modulefile_toast.in"

out="modver_toast-${version}"
rm -f ${out}

while IFS='' read -r line || [[ -n "${line}" ]]; do
    echo "${line}" | eval sed ${sub} >> "${out}"
done < "${topdir}/version.in"

echo "run configure with --prefix=${prefix}/toast/${version}"

