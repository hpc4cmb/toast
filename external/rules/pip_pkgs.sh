# NOTE: for Docker images, this should be called after MPI
#     installation since TiMemory will not include MPI support
#     if this is run before MPI is installed
pip install --no-binary :all: \
    fitsio \
    timemory \
    pyinstaller
