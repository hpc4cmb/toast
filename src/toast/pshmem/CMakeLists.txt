
# Install the python files

install(FILES
    __init__.py
    locking.py
    shmem.py
    utils.py
    test.py
    DESTINATION ${PYTHON_SITE}/toast/pshmem
)
