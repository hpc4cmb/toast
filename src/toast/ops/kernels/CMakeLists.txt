
# Install the python files

install(FILES
    __init__.py
    implementation_selection.py
    DESTINATION ${PYTHON_SITE}/toast/ops/kernels
)

add_subdirectory(jax_kernels)
add_subdirectory(python_kernels)
add_subdirectory(compiled_kernels)
