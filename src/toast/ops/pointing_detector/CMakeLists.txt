
# Install the python files

install(FILES
    __init__.py
    pointing_detector.py
    kernels.py
    kernels_numpy.py
    kernels_jax.py
    DESTINATION ${PYTHON_SITE}/toast/ops/pointing_detector
)
