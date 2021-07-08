#!/bin/bash
#
# Apply TOAST source code formatting.  This script will run the
# "uncrustify" and "black" formatting tools on C++ and Python code
# respectively.
#

# Get the directory containing this script
pushd $(dirname $0) > /dev/null
base=$(pwd -P)
popd > /dev/null

# Check executables
unexe=$(which uncrustify)
if [ "x${unexe}" = "x" ]; then
    echo "Cannot find the \"uncrustify\" executable.  Is it in your PATH?"
    exit 1
fi

umajor=$(uncrustify --version | sed -e "s#.*-\([0-9]\+\)\.[0-9]\+\.[0-9]\+_.*#\1#")
uminor=$(uncrustify --version | sed -e "s#.*-[0-9]\+\.\([0-9]\+\)\.[0-9]\+_.*#\1#")
upatch=$(uncrustify --version | sed -e "s#.*-[0-9]\+\.[0-9]\+\.\([0-9]\+\)_.*#\1#")

echo "Found uncrustify version ${umajor}.${uminor}.${upatch}"
if [ ${umajor} -eq "0" ]; then
    if [ ${uminor} -lt "73" ]; then
        echo "This script requires at least uncrustify version 0.73.0"
        exit 1
    fi
fi

blkexe=$(which black)
if [ "x${blkexe}" = "x" ]; then
    echo "Cannot find the \"black\" executable.  Is it in your PATH?"
    exit 1
fi

bmajor=$(black --version | awk '{print $3}' | sed -e "s#\([0-9]\+\)\.[0-9]\+.*#\1#")
bminor=$(black --version | awk '{print $3}' | sed -e "s#[0-9]\+\.\([0-9]\+\).*#\1#")

echo "Found black version ${bmajor}.${bminor}"
if [ ${bmajor} -le "21" ]; then
    if [ ${bminor} -lt "5" ]; then
        echo "This script requires at least black version 21.5"
        exit 1
    fi
fi

# The uncrustify config file
uncfg="${base}/uncrustify.cfg"

# Uncrustify runtime options per file
unrun="-c ${uncfg} --replace --no-backup"

# Uncrustify test options
untest="-c ${uncfg} --check"

# Black runtime options
blkrun="-l 88"

# Black test options
blktest="--check"

# Process directories with C++ files
for cppd in libtoast toast; do
    find "${base}/${cppd}" -name "*.hpp" -not -path '*Random123*' -not -path '*pybind11/*' -not -path '*gtest/*' -exec echo "uncrustify {}" \; -exec ${unexe} ${unrun} '{}' \;
    find "${base}/${cppd}" -name "*.cpp" -not -path '*Random123*' -not -path '*pybind11/*' -not -path '*gtest/*' -exec echo "uncrustify {}" \; -exec ${unexe} ${unrun} '{}' \;
done

# Process directories with *.py files
for pyd in toast ../workflows; do
    find "${base}/${pyd}" -name "*.py" -not -path '*pybind11/*' -exec echo "Formatting {}" \; -exec ${blkexe} ${blkrun} '{}' \;
done

# Special case:  process files in the scripts directory which do not
# have the .py extension
for pyd in toast/scripts; do
    find "${base}/${pyd}" -name "toast_*" -exec echo "Formatting {}" \; -exec ${blkexe} ${blkrun} '{}' \;
done
