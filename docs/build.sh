#!/bin/bash

# Runs zensical build and then updates notebooks in the site dir

rm -rf ./site
zensical build
./convert_zensical_notebooks.py ./docs ./site
