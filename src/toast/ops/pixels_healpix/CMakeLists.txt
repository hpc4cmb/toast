
# Install the python files

install(FILES
    __init__.py
    pixels_healpix.py
    kernels.py
    kernels_numpy.py
    kernels_jax.py
    DESTINATION ${PYTHON_SITE}/toast/ops/pixels_healpix
)
