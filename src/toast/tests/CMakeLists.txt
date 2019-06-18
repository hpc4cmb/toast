
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
    DESTINATION ${PYTHON_SITE}/toast/tests
)
