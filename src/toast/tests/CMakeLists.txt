
# Install the python files

install(FILES
    __init__.py
    _helpers.py
    runner.py
    mpi.py
    dist.py
    cache.py
    rng.py
    qarray.py
    fft.py
    DESTINATION ${PYTHON_SITE}/toast/tests
)
