bash fetch_data.sh > /dev/null 2>&1
bash generate_shell.sh
# nside
sed -i "s/512/64/g" tiny* params/satellite/sim_noise_hwp.par
# zip -> skip_atmo in ground
sed -i "s/zip/skip_atmosphere/" params/ground/*par
# just make 30 madam iterations in ground, we don't test destriped maps
# make sure that file doesn't contain madam_iter_max already so we
# avoid applying this twice
for file in params/ground/*par
do
    if ! grep -q "madam_iter_max" $file; then
        sed -i "s/--madam/--madam_iter_max\n30\n--madam/" $file
    fi
done
# duration
sed -i "s/24/1/g" tiny*
# fake focalplane disable mpi
sed -i "s/mpirun -n 1//g" tiny*
# write log to stdout
sed -i 's/eval \${run} \${com}.*$/eval \${run} \${com}/' tiny*
# 2 procs, 1 thread each
sed -i 's/OMP_NUM_THREADS=\${threads}/OMP_NUM_THREADS=1/' tiny*

: ${TYPES:="satellite ground ground_simple ground_multisite"}
exit_status=0

for TYPE in $TYPES
do
    echo ">>>>>>>>>> Running test for $TYPE"
    wget https://github.com/hpc4cmb/toast-test-data/blob/master/examples/ref_out_tiny_${TYPE}.tgz?raw=true > /dev/null 2>&1
    tar xzf ref_out_tiny_${TYPE}.tgz > /dev/null 2>&1
    bash tiny_${TYPE}_shell.sh && python check_maps.py $TYPE; (( exit_status = exit_status || $? ))
done
exit $exit_status
