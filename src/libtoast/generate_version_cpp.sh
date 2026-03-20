#!/bin/sh

cmake_version=$1

c_file="version.cpp"
echo "Using file ${c_file}"
echo "cmake version = ${cmake_version}"
c_version=""
if [ -f "${c_file}" ]; then
    # Existing file, read in version
    c_version=`cat "${c_file}" | grep TOAST_VERSION | sed -e 's/.*= "\(.*\)".*/\1/'`
fi

echo "Got C version ${c_version}"

if [ "x${c_version}" != "x${cmake_version}" ]; then
    # The version has changed, update the file
    echo "UPDATING"
    echo '// This file is auto-generated, do not edit.' > "${c_file}"
    echo "const char* TOAST_VERSION = \"${cmake_version}\";" >> "${c_file}"
fi
