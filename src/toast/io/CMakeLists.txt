
# Install the python files

install(FILES
    __init__.py
    observation_hdf_save.py
    observation_hdf_load.py
    observation_hdf_load_v0.py
    hdf_utils.py
    compression.py
    compression_flac.py
    DESTINATION ${PYTHON_SITE}/toast/io
)
