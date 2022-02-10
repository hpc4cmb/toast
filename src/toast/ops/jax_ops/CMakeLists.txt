
# Install the python files

install(FILES
    __init__.py
    polyfilter1D.py
    polyfilter2D.py
    scan_map.py
    healpix_pixels.py
    stokes_weights.py
    cov_accum.py
    template_offset.py
    DESTINATION ${PYTHON_SITE}/toast/ops/jax_ops
)

add_subdirectory(utils)
