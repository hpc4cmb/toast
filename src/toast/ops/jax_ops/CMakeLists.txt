
# Install the python files

install(FILES
    __init__.py
    polyfilter1D.py
    polyfilter2D.py
    scan_map.py
    pixels_healpix.py
    stokes_weights.py
    template_offset.py
    qarray.py
    pointing_detector.py
    build_noise_weighted.py
    DESTINATION ${PYTHON_SITE}/toast/ops/jax_ops
)

add_subdirectory(utils)
