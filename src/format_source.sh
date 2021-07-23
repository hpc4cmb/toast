#!/bin/bash
#
# Apply TOAST source code formatting.  This script will run the
# "uncrustify" and "black" formatting tools on C++ and Python code
# respectively.
#

# Get the directory containing this script
base="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
blkrun="-l 88"

# Black test options
blktest="--check"

cd "$base"
# Directories to process
find libtoast toast -name "*.hpp" -not -path '*Random123*' -not -path '*pybind11/*' -not -path '*gtest/*' -exec ${unexe} ${unrun} {} + &
find toast ../pipelines -name "*.py" -not -path '*pybind11/*' -exec ${blkexe} ${blkrun} {} + &
wait
