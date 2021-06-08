
# Install the python files

install(FILES
    __init__.py
    applygain.py
    crosstalk.py
    gainscrambler.py
    interval.py
    memorycounter.py
    noise.py
    polyfilter.py
    sim_det_noise.py
    sim_focalplane.py
    sim_interval.py
    sim_noise.py
    spt3g_utils.py
    spt3g.py
    tidas_utils.py
    tidas.py
    tod_math.py
    tod.py
    spt3g_utils.py
    tidas_utils.py
    tidas.py
    DESTINATION ${PYTHON_SITE}/toast/tod
)
