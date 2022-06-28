
# Install the python files

install(FILES
    __init__.py
    math_qarray.py
    math_healpix.py
    intervals.py
    mutableArray.py
    asserts.py
    DESTINATION ${PYTHON_SITE}/toast/ops/jax_ops/utils
)
