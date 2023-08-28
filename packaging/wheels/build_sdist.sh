#!/bin/bash
#
# This script is designed to run within the toast dependency docker container in order
# to build the source distribution.
#

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
topdir=$(pwd)
popd >/dev/null 2>&1

# The toast source tree is mounted at /home/toast

cp -a /home/toast .

pushd toast

mkdir -p dist
rm -f dist/*

python setup.py sdist
chown $(id -u):$(id -g) dist/*

cp -a dist/toast* /home/toast/dist/

popd
