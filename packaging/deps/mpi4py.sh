# Variables set by calling code:
# - MPICC (optional)
#

# Install mpi4py

echo "Installing mpi4py..."
if [ "x${MPICC}" = "x" ]; then
    echo "The MPICC environment variable is not set.  Installing mpi4py"
    echo "from the conda package, rather than building from source."
    conda install --yes mpich mpi4py
else
    echo "Building mpi4py with MPICC=\"${MPICC}\""
    pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
fi

