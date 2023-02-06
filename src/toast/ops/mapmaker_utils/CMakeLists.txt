
# Install the python files

install(FILES
    __init__.py
    mapmaker_utils.py
    kernels.py
    kernels_numpy.py
    kernels_jax.py
    DESTINATION ${PYTHON_SITE}/toast/ops/mapmaker_utils
)
