bash fetch_data.sh
bash generate_shell.sh
# nside
sed -i "s/512/64/g" tiny* params/satellite/sim_noise_hwp.par
# zip -> skip_atmo in ground
sed -i "s/zip/skip_atmosphere/" params/ground/*par
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
    wget http://portal.nersc.gov/project/cmb/toast_data/ref_out_tiny_${TYPE}.tgz > /dev/null 2>&1
    tar xzf ref_out_tiny_${TYPE}.tgz > /dev/null 2>&1
    bash tiny_${TYPE}_shell.sh && python check_maps.py $TYPE; (( exit_status = exit_status || $? ))
done
exit $exit_status
