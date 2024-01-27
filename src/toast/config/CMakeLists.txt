
# Install the python files

install(FILES
    __init__.py
    cli.py
    utils.py
    json.py
    toml.py
    yaml.py
    DESTINATION ${PYTHON_SITE}/toast/config
)
