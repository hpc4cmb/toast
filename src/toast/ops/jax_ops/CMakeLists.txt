
# Install the python files

install(FILES
    __init__.py
    utils.py
    polyfilter1D.py
    polyfilter2D.py
    scan_map.py
    DESTINATION ${PYTHON_SITE}/toast/ops/jax_ops
)
