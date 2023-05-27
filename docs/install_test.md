(install:test)=
# Testing the Installation

After installation (regardless of method), you can run both the compiled and
python unit tests.  These tests will create an output directory named
`toast_test_output` in your current working directory:

```{code-block} console
python -c "import toast.tests; toast.tests.run()"
```

If you have installed the `mpi4py` package, then you can also run the unit tests with
MPI enabled.  For example:

```{code-block} console
export OMP_NUM_THREADS=2
mpirun -np 2 python -c "import toast.tests; toast.tests.run()"
```

```{important}
You should use whatever MPI launcher is appropriate for your system (e.g.  `mpirun`,
`mpiexec`, `srun`, etc).  In general, be sure to set the `OMP_NUM_THREADS` environment
variable so that the number of MPI processes times this number of threads is not greater
than the number of physical CPU cores.
```

The runtime configuration of toast can also be checked with an included script:

```{code-block} bash
toast_env
```
