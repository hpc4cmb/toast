
# Install the python files

install(FILES
  __init__.py
  atm.py
  dist.py
  export.py
  filters.py
  gain.py
  madam.py
  noise.py
  pointing.py
  sky_signal.py
  sss.py
  todground.py
  DESTINATION ${PYTHON_SITE}/toast/pipeline_tools
)
