
# Install the python files

install(FILES
    __init__.py
    scan_map.py
    kernels.py
    kernels_numpy.py
    kernels_jax.py
    DESTINATION ${PYTHON_SITE}/toast/ops/scan_map
)
