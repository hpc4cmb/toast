# Variables set by calling code:
# - MPICC (optional)
#

# Install mpi4py

echo "Installing mpi4py..."
if [ "x${MPICC}" = "x" ]; then
    echo "The MPICC environment variable is not set.  Installing mpi4py"
    echo "from the conda package, rather than building from source."
    if [ "x$(which conda)" = "x" ]; then
        echo "Conda not available- giving up!"
    else
        conda install --yes mpich mpi4py
    fi
else
    echo "Building mpi4py with MPICC=\"${MPICC}\""
    CFLAGS="-O2 -g -fPIC" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
fi

