# Installation

TOAST is written in C++ and python3 and depends on several commonly
available packages. It also has some optional functionality that is only
enabled if additional external packages are available. The best
installation method will depend on your specific needs. We try to
clarify the different options in the following sections.


(install:test)=
## Testing the Installation

After installation (regardless of how you did that), you can run both the compiled and python unit tests.
These tests will create an output directory named `toast_test_output` in your current
working directory:

```{code-block} console
python -c "import toast.tests; toast.tests.run()"
```

If you have installed the `mpi4py` package, then you can also run the unit tests with MPI enabled.  For example:

```{code-block} console
mpirun -np 4 python -c "import toast.tests; toast.tests.run()"
```

```{important}
You should use whatever MPI launcher is appropriate for your system (e.g. `mpirun`, `mpiexec`, `srun`, etc).  In general, be sure to set the `OMP_NUM_THREADS` environment variable so that the number of MPI processes times this number of threads is not greater than the number of physical CPU cores.
```

The runtime configuration of toast can also be checked with an included
script:

```{code-block} bash
toast_env
```
