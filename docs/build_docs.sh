#!/bin/bash

# Get the absolute path to the folder with this script
pushd $(dirname $0) > /dev/null
docdir=$(pwd -P)
popd > /dev/null

pushd ${docdir} > /dev/null

./setup_docs.sh

jupyter-book build .

popd > /dev/null
