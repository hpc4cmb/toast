
# Install the python files

install(FILES
    __init__.py
    observation_hdf_save.py
    observation_hdf_save_v1.py
    observation_hdf_load.py
    observation_hdf_load_v0.py
    observation_hdf_load_v1.py
    hdf_utils.py
    deprecated_compression.py
    DESTINATION ${PYTHON_SITE}/toast/io
)
