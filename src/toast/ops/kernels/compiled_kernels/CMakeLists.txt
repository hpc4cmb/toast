
# Install the python files

install(FILES
    __init__.py
    scan_map.py
    noise_weight.py
    DESTINATION ${PYTHON_SITE}/toast/ops/kernels/compiled_kernels
)
