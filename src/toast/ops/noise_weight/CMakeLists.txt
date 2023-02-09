
# Install the python files

install(FILES
    __init__.py
    noise_weight.py
    kernels.py
    kernels_numpy.py
    kernels_jax.py
    DESTINATION ${PYTHON_SITE}/toast/ops/noise_weight
)
