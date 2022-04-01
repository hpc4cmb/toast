
# Install the python files

install(FILES
    __init__.py
    observation_hdf_save.py
    observation_hdf_load.py
    hdf_utils.py
    DESTINATION ${PYTHON_SITE}/toast/io
)
