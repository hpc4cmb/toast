
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
    healpix.py
    DESTINATION ${PYTHON_SITE}/toast/tests
)
