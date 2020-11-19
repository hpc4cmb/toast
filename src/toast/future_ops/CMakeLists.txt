
# Install the python files

install(FILES
    __init__.py
    pipeline.py
    memory_counter.py
    sim_hwp.py
    sim_tod_noise.py
    sim_satellite.py
    noise_model.py
    pointing_healpix.py
    mapmaker_utils.py
    DESTINATION ${PYTHON_SITE}/toast/future_ops
)
