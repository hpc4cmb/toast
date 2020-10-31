
# Install the python files

install(FILES
    __init__.py
    _helpers.py
    runner.py
    mpi.py
    env.py
    dist.py
    intervals.py
    rng.py
    qarray.py
    fft.py
    healpix.py
    config.py
    observation.py
    timing.py
    pixels.py
    ops_sim_satellite.py
    ops_applygain.py
    ops_sim_tod_noise.py
    covariance.py
    ops_pmat.py
    ops_dipole.py
    ops_groundfilter.py
    sim_focalplane.py
    ops_polyfilter.py
    ops_memory_counter.py
    ops_gainscrambler.py
    psd_math.py
    ops_madam.py
    ops_mapmaker.py
    map_satellite.py
    map_ground.py
    binned.py
    ops_sim_pysm.py
    tidas.py
    ops_sim_atm.py
    ops_sim_sss.py
    DESTINATION ${PYTHON_SITE}/toast/tests
)
