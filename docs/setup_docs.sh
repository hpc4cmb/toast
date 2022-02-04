#!/bin/bash

# Get the absolute path to the folder with this script
pushd $(dirname $0) > /dev/null
docdir=$(pwd -P)
popd > /dev/null

pushd ${docdir} > /dev/null

rm -rf _build conf.py

jupyter-book config sphinx .

echo "" >> conf.py
echo "import toast" >> conf.py
echo "import toast.ops" >> conf.py

popd > /dev/null
