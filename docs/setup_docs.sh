#!/bin/bash

# Get the absolute path to the folder with this script
pushd $(dirname $0) > /dev/null
docdir=$(pwd -P)
popd > /dev/null

pushd ${docdir} > /dev/null

rm -rf _build conf.py

# Generate code blocks for scripts
cli_names="\
toast_ground_schedule \
toast_satellite_schedule \
toast_benchmark_satellite \
toast_benchmark_ground \
toast_benchmark_ground_setup \
"

for cli in ${cli_names}; do
    echo "Generating usage info for ${cli}"
    eval "${docdir}/create_help_inc.sh" ${cli} > "${docdir}/${cli}.inc"
done

jupyter-book config sphinx .

echo "" >> conf.py
echo "import toast" >> conf.py
echo "import toast.ops" >> conf.py

popd > /dev/null
