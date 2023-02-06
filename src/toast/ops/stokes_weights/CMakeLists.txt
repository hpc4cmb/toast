
# Install the python files

install(FILES
    __init__.py
    stokes_weights.py
    kernels.py
    kernels_numpy.py
    kernels_jax.py
    DESTINATION ${PYTHON_SITE}/toast/ops/stokes_weights
)
