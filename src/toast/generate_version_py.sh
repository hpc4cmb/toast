#!/bin/sh

cmake_version=$1

py_file="_version.py"
echo "Using file ${py_file}"
py_version=""
if [ -f "${py_file}" ]; then
    # Existing file, read in version
    py_version=`cat "${py_file}" | sed -e 's/.*= "\(.*\)".*/\1/'`
fi

echo "Got py version ${py_version}"

if [ "x${py_version}" != "x${cmake_version}" ]; then
    # The version has changed, update the file
    echo "UPDATING"
    echo "__version__ = \"${cmake_version}\"" > "${py_file}"
fi
