
# Install the python files

install(FILES
    __init__.py
    polyfilter1D.py
    polyfilter2D.py
    template_offset.py
    noise_weight.py
    DESTINATION ${PYTHON_SITE}/toast/ops/kernels/python
)

add_subdirectory(math)
