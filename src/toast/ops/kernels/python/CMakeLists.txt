
# Install the python files

install(FILES
    __init__.py
    DESTINATION ${PYTHON_SITE}/toast/ops/kernels/python
)

add_subdirectory(math)
