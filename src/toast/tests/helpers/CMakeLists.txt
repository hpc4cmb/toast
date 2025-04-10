
# Install the python files

install(FILES
    __init__.py
    flags.py
    ground.py
    hwp.py
    sky.py
    space.py
    utils.py
    DESTINATION ${PYTHON_SITE}/toast/tests/helpers
)
