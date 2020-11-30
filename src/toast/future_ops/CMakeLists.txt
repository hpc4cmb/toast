
# Install the python files

install(FILES
    __init__.py
    pipeline.py
    clear.py
    copy.py
    memory_counter.py
    sim_hwp.py
    sim_tod_noise.py
    sim_satellite.py
    noise_model.py
    noise_weight.py
    pointing_healpix.py
    scan_map.py
    mapmaker_utils.py
    mapmaker_binning.py
    DESTINATION ${PYTHON_SITE}/toast/future_ops
)
