bash generate_shell.sh
sed -i "s/512/64/g" tiny*
wget http://portal.nersc.gov/project/cmb/toast_data/ref_out_tiny_satellite.tgz
tar xzf ref_out_tiny_satellite.tgz
bash tiny_satellite_shell.sh
python check_maps.py
