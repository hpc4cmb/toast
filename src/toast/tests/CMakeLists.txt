
# Install the python files

install(FILES
    __init__.py
    _helpers.py
    runner.py
    mpi.py
    dist.py
    timing.py
    cache.py
    rng.py
    qarray.py
    fft.py
    tod.py
    intervals.py
    tod_satellite.py
    ops_applygain.py
    ops_simnoise.py
    cov.py
    ops_pmat.py
    ops_dipole.py
    ops_groundfilter.py
    sim_focalplane.py
    ops_polyfilter.py
    ops_memorycounter.py
    ops_gainscrambler.py
    psd_math.py
    ops_madam.py
    map_satellite.py
    map_ground.py
    binned.py
    ops_sim_pysm.py
    smooth.py
    DESTINATION ${PYTHON_SITE}/toast/tests
)
