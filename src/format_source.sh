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

blkexe=$(which black)
if [ "x${blkexe}" = "x" ]; then
    echo "Cannot find the \"black\" executable.  Is it in your PATH?"
    exit 1
fi

# The uncrustify config file
uncfg="${base}/uncrustify.cfg"

# Uncrustify runtime options per file
unrun="-c ${uncfg} --replace --no-backup"

# Uncrustify test options
untest="-c ${uncfg} --check"

# Black runtime options
blkrun=""

# Black test options
blktest="--check"

# Directories to process
cppdirs="libtoast/src libtoast/include libtoast_mpi/src libtoast_mpi/include"
pydirs="toast"

# Test
for cppd in ${cppdirs}; do
    find "${base}/${cppd}" -name "*.hpp" -exec ${unexe} ${untest} '{}' \;
    find "${base}/${cppd}" -name "*.cpp" -exec ${unexe} ${untest} '{}' \;
done

for pyd in ${pydirs}; do
    find "${base}/${pyd}" -name "*.py" -exec ${blkexe} ${blktest} '{}' \;
done
