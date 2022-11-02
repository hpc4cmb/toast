
# Install the python files

install(FILES
    __init__.py
    intervals.py
    mutableArray.py
    data_localization.py
    gpu_selection.py 
    implementation_selection.py
    DESTINATION ${PYTHON_SITE}/toast/jax
)
