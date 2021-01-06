
# Install the python files

install(FILES
    __init__.py
    atm.py
    conviqt.py
    groundfilter.py
    madam.py
    mapmaker.py
    mapsampler.py
    pointing_math.py
    pointing.py
    pysm.py
    sim_det_atm.py
    sim_det_dipole.py
    sim_det_map.py
    sim_det_pysm.py
    sim_tod.py
    sss.py
    todmap_math.py
    DESTINATION ${PYTHON_SITE}/toast/todmap
)
