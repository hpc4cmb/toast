
# Install the python files

install(FILES
    __init__.py
    polyfilter1D.py
    polyfilter2D.py
    scan_map.py
    stokes_weights.py
    template_offset.py
    build_noise_weighted.py
    cov_accum.py
    noise_weight.py
    DESTINATION ${PYTHON_SITE}/toast/ops/kernels/python
)

add_subdirectory(math)
