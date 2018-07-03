#!/bin/sh

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

TOASTDATACOMMIT="294782673daaf7328b77977296c8334888c9f65a"

bash fetch_data.sh > /dev/null 2>&1
bash generate_shell.sh
# nside
sed -i.bak "s/512/64/g" tiny* params/satellite/sim_noise_hwp.par
sed -i.bak "/zip/d" params/ground/ground_sim_simple.par
# zip -> skip_atmo in ground
sed -i.bak "s/zip/skip_atmosphere/" params/ground/ground_sim.par params/ground/ground_sim_multisite.par
# just make 30 madam iterations in ground, we don't test destriped maps
# make sure that file doesn't contain madam_iter_max already so we
# avoid applying this twice
for file in params/ground/*par
do
    if ! grep -q "madam_iter_max" $file; then
        sed -i.bak "s/--madam/--madam_iter_max\n30\n--madam/" $file
    fi
done
# duration
sed -i.bak "s/24/1/g" tiny*
# fake focalplane disable mpi
sed -i.bak "s/mpirun -n 1//g" tiny*
# write log to stdout
sed -i.bak 's/eval \${run} \${com}.*$/eval \${run} \${com}/' tiny*
# 2 procs, 1 thread each
sed -i.bak 's/OMP_NUM_THREADS=\${threads}/OMP_NUM_THREADS=1/' tiny*

find . -name "*.bak" -delete

: ${TYPES:="satellite ground ground_simple ground_multisite"}
exit_status=0

for TYPE in $TYPES
do
    echo ">>>>>>>>>> Running test for $TYPE"
    # uncomment this to automatically pickup the latest version
    # wget --output-document=ref_out_tiny_${TYPE}.tgz https://github.com/hpc4cmb/toast-test-data/blob/master/examples/ref_out_tiny_${TYPE}.tgz?raw=true > /dev/null 2>&1
    wget --output-document=ref_out_tiny_${TYPE}.tgz https://github.com/hpc4cmb/toast-test-data/blob/${TOASTDATACOMMIT}/examples/ref_out_tiny_${TYPE}.tgz?raw=true > /dev/null 2>&1
    tar xzf ref_out_tiny_${TYPE}.tgz > /dev/null 2>&1
    bash tiny_${TYPE}_shell.sh && python check_maps.py $TYPE; (( exit_status = exit_status || $? ))
done
exit $exit_status
