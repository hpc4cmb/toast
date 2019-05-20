
# Install the python files

install(FILES
    __init__.py
    applygain.py
    conviqt.py
    gainscrambler.py
    groundfilter.py
    interval.py
    memorycounter.py
    noise.py
    pointing_math.py
    pointing.py
    polyfilter.py
    sim_det_atm.py
    sim_det_dipole.py
    sim_det_map.py
    sim_det_noise.py
    sim_focalplane.py
    sim_interval.py
    sim_noise.py
    sim_tod.py
    spt3g_utils.py
    spt3g.py
    tidas_utils.py
    tidas.py
    tod_math.py
    tod.py
    DESTINATION ${PYTHON_SITE}/toast/tod
)
