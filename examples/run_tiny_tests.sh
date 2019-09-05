#!/bin/bash

# Abort on Error
set -e

# How to update the test maps

# If a change in the TOAST code results in different maps for one of more of the integration tests
# it is necessary to update them. How to do it:

# * run `run_tiny_tests.sh` on your machine, you can set the `TYPES` environment variable to run only a subset of the tests
# * this will download the current expected test results and untar them in `ref_out_tiny_${TYPE} folders
# * the script will produce outputs in `out_tiny_${TYPE}`
# * compare interactively the results in the 2 folders and make sure the differences are expected
# * now remove all the unnecessary files from `out_tiny_${TYPE}` so that it only contains the files in `ref_out_tiny_${TYPE}`
# * tar the `ref` folder to tgz and upload to: https://github.com/hpc4cmb/toast-test-data/tree/master/examples
# * updated `TOASTDATACOMMIT` below to the latest commit
# * run `run_tiny_tests.sh` again and check that the test passes

TOASTDATACOMMIT=d50dfea8a1d939bfc4681171198f5c31eed5fee7

if [ "x${TYPES}" = "x" ]; then
    TYPES="satellite ground ground_simple ground_multisite"
fi

bash fetch_data.sh > /dev/null 2>&1
bash generate_shell.sh

# Adjust the shell scripts to run quicker
for f in tiny*sh; do
    # nside
    sed -i.bak "s/512/64/g" $f
    # duration
    sed -i.bak "s/24/1/g" $f
    # fake focalplane disable mpi
    sed -i.bak "s/mpirun -n 1//g" $f
    # write log to stdout
    sed -i.bak 's/eval \${run} \${com}.*$/eval \${run} \${com}/' $f
    # 2 procs, 1 thread each
    sed -i.bak 's/OMP_NUM_THREADS=\${threads}/OMP_NUM_THREADS=1/' $f
done
# Do not zip the binned maps
if [ -e tiny_satellite_shell.sh ]; then
    sed -i.bak "s/\$\@/--no-zip $\@/g" tiny_satellite_shell.sh
fi
if [ -e tiny_ground_simple_shell.sh ]; then
    sed -i.bak "s/\$\@/--no-zip $\@/g" tiny_ground_simple_shell.sh
fi
# Skip the atmospheric simulation, limit madam to 30 iterations
if [ -e tiny_ground_shell.sh ]; then
    sed -i.bak "s/\$\@/--no-atmosphere $\@/g" tiny_ground_shell.sh
    sed -i.bak "s/\$\@/--madam-iter-max 30 $\@/g" $f
fi
if [ -e tiny_ground_simple_shell.sh ]; then
    sed -i.bak "s/\$\@/--madam-iter-max 30 $\@/g" $f
fi
if [ -e tiny_ground_multisite_shell.sh ]; then
    sed -i.bak "s/\$\@/--no-atmosphere $\@/g" tiny_ground_multisite_shell.sh
    sed -i.bak "s/\$\@/--madam-iter-max 30 $\@/g" $f
fi

find . -name "*.bak" -delete

for TYPE in ${TYPES}; do
    echo ">>>>>>>>>> Running test for $TYPE"
    # uncomment this to automatically pickup the latest version
    # wget --output-document=ref_out_tiny_${TYPE}.tgz https://github.com/hpc4cmb/toast-test-data/blob/master/examples/ref_out_tiny_${TYPE}.tgz?raw=true > /dev/null 2>&1
    wget --output-document=ref_out_tiny_${TYPE}.tgz https://github.com/hpc4cmb/toast-test-data/blob/${TOASTDATACOMMIT}/examples/ref_out_tiny_${TYPE}.tgz?raw=true > /dev/null 2>&1
    tar xzf ref_out_tiny_${TYPE}.tgz > /dev/null 2>&1

    bash tiny_${TYPE}_shell.sh \
    && python check_maps.py $TYPE
done
