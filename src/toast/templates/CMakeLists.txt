
# Install the python files

install(FILES
    __init__.py
    template.py
    amplitudes.py
    fourier2d.py
    subharmonic.py
    gaintemplate.py
    DESTINATION ${PYTHON_SITE}/toast/templates
)

add_subdirectory(offset)
