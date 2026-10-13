#!/bin/bash

# Runs zensical build and then updates notebooks in the site dir

rm -rf ./site
# Use the wrapper script to pre-import toast in the main thread (required for
# the dynamic API docs, see zensical_build.py)
./zensical_build.py
./convert_zensical_notebooks.py ./docs ./site
