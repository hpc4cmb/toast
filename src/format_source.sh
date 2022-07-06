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

umajor=$(uncrustify --version | sed -Ee 's#.*-([0-9]+)\.([0-9]+)\.([0-9]+).*#\1#')
uminor=$(uncrustify --version | sed -Ee 's#.*-([0-9]+)\.([0-9]+)\.([0-9]+).*#\2#')
upatch=$(uncrustify --version | sed -Ee 's#.*-([0-9]+)\.([0-9]+)\.([0-9]+).*#\3#')

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

isortexe=$(which isort)
have_isort="yes"
if [ "x${isortexe}" = "x" ]; then
    echo "Cannot find the \"isort\" executable.  Is it in your PATH?"
    echo "Skipping for now."
    have_isort="no"
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

# Note that the "+" argument to "find ... -exec" below passes all found files to the
# exec command in one go.  This works because both uncrustify and black accept multiple
# files as arguments.

# Process directories with C++ files.
find "${base}/libtoast" "${base}/toast" \( -name "*.hpp" -or -name "*.cpp" \) \
    -and -not \( -path '*Random123*' -or -path '*pybind11/*' -or -path '*gtest/*' \) \
    -exec ${unexe} ${unrun} '{}' + &

# Process directories with python files
find "${base}/toast" "${base}/../workflows" -name "*.py" -and -not \
    -path '*pybind11/*' -exec ${blkexe} ${blkrun} '{}' + &
if [ ${have_isort} = "yes" ]; then
    find "${base}/toast" "${base}/../workflows" -name "*.py" -and -not \
    -path '*pybind11/*' -exec ${isortexe} --profile black '{}' + &
fi

# Special case:  process files in the scripts directory which do not
# have the .py extension
find "${base}/toast/scripts" -name "toast_*" -exec ${blkexe} ${blkrun} '{}' + &
if [ ${have_isort} = "yes" ]; then
    find "${base}/toast/scripts" -name "toast_*" -exec ${isortexe} --profile black '{}' + &
fi

# Wait for the commands to finish
wait
