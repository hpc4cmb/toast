
# Install the python files

install(FILES
    __init__.py
    implementation_selection.py
    DESTINATION ${PYTHON_SITE}/toast/ops/kernels
)

add_subdirectory(jax)
add_subdirectory(python)
add_subdirectory(compiled)
