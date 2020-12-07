
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
    ops_sim_tod_noise.py
    ops_mapmaker_utils.py
    covariance.py
    ops_pointing_healpix.py
    sim_focalplane.py
    ops_memory_counter.py
    ops_scan_map.py
    DESTINATION ${PYTHON_SITE}/toast/tests
)
