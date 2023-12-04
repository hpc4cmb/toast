
# Install the python files

install(FILES
    __init__.py
    intervals.py
    maps.py
    mutableArray.py
    device.py 
    DESTINATION ${PYTHON_SITE}/toast/jax
)

add_subdirectory(math)
