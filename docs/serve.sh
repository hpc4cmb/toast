#!/bin/bash

set -m

rm -rf ./site
zensical serve &

echo "Waiting for site build..."
sleep 10

echo "Converting notebooks..."
./convert_zensical_notebooks.py ./docs ./site

echo "Returning control to zensical serve..."
fg %1
