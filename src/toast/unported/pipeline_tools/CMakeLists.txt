
# Install the python files

install(FILES
  __init__.py
  atm.py
  binning.py
  debug.py
  dipole.py
  dist.py
  export.py
  filters.py
  gain.py
  mapmaker.py
  madam.py
  noise.py
  pointing.py
  sky_signal.py
  sss.py
  todground.py
  todsatellite.py
  DESTINATION ${PYTHON_SITE}/toast/pipeline_tools
)
