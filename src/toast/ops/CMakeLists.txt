
# Install the python files

install(FILES
    __init__.py
    operator.py
    pipeline.py
    delete.py
    copy.py
    reset.py
    arithmetic.py
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
    mapmaker_solve.py
    mapmaker_templates.py
    mapmaker.py
    madam.py
    madam_utils.py
    DESTINATION ${PYTHON_SITE}/toast/ops
)
