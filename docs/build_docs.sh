#!/bin/bash

# Copy the docs directory to the local working directory and build it with sphinx

set -e

# Get the absolute path to the docs folder
pushd $(dirname $0) > /dev/null
docdir=$(pwd -P)
popd > /dev/null

cp -a "${docdir}" ./docs
pushd ./docs > /dev/null
make html
popd > /dev/null
