(install:dev)=
# Developer Installation

If you wish the modify the TOAST source or test different build options, then you will need to compile from source.  Here we will discuss several specific system configurations that are known to work. The best one for you will depend on your OS and preferences.

The required dependencies for building TOAST from source include:

- A C/C++ compiler
- CMake >= 3.20
- BLAS / LAPACK libraries
- FFT library (FFTW or MKL)
- Python >= 3.7 and various python packages

Additionally, some features in TOAST will become available if external packages are importable at runtime:

Feature | Runtime Requirement
--------|------------------------------
MPI parallelism | [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html)
Atmosphere simulation | SuiteSparse and [libaatm](https://github.com/hpc4cmb/libaatm)
MADAM mapmaking operator | [libmadam](https://github.com/hpc4cmb/libmadam)
Conviqt beam convolution operator | [libconviqt](https://github.com/hpc4cmb/libconviqt)
Totalconvolve beam convolution operator | [ducc0](https://pypi.org/project/ducc0)

## Installing Build Dependencies

The compiled extension within TOAST has several libraries that need to be found at build time.
Most Linux distributions have all the required dependencies of TOAST available in their package repositories.  On MacOS you can meet these dependencies with homebrew or macports.  Here we provide a couple of examples.

### Debian / Ubuntu

On recent versions of debian-based distros you can get all required build dependencies and some optional ones from OS packages:

```{code-block} bash
apt update
apt install \
    cmake \
    build-essential \
    libopenblas-dev \
    libmpich-dev \
    liblapack-dev \
    libfftw3-dev \
    libsuitesparse-dev \
    python3-dev \
    libpython3-dev
```

```{note}
If you are using another package on your system that requires
OpenMPI, then you may get a conflict installing libmpich-dev. In that
case, just install libopenmpi-dev instead.
```

To enable support for atmosphere simulations, download a [release of
libaatm](https://github.com/hpc4cmb/libaatm/releases) and install it.
For example:

    cd libaatm
    mkdir build
    cd build
    cmake \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        ..
    make -j 4
    sudo make install

You can also install it to the same prefix as TOAST or to a separate
location for just the TOAST dependencies. If you install it somewhere
other than /usr/local then make sure it is in your environment search
paths (see the \"installing TOAST\" section).

You can also now install the optional dependencies if you wish:

> -   [libconviqt](https://github.com/hpc4cmb/libconviqt) for 4PI beam
>     convolution.  You will need a C++ MPI compiler.
> -   [libmadam](https://github.com/hpc4cmb/libmadam) for optimized
>     destriping mapmaking.  You will need a Fortran MPI compiler.


### OS X with MacPorts

```{admonition} To-Do
Document using macports to get gcc and installing optional dependencies.
```

### OS X with Homebrew

```{admonition} To-Do
Document installing compiled dependencies and using a virtualenv.
```

### Conda

The conda-forge channel provides all the necessary build dependencies for TOAST, so this is another avenue to meet those requirements [see the conda section for setting up an environment](install:user:conda).

```{warning}
Conda packages contain libraries that are likely not consistent with software installed by your OS.  If you want to use conda to provide those libraries for your build, you **must** install and use the conda-provided gcc toolchain.
```

If you are using conda only to get the python interpreter and python packages (not any compile-time dependencies), then you should be aware that the conda toolchain installs its own version of the linker (`ld`) which can interfere with using other compiler toolchains.  If you are not building anything with the conda compilers, it is safe to remove this:

```{code-block} bash
rm -f $(dirname $(dirname $(which python)))/compiler_compat/ld
```

## Custom Install

You can also do something completely different and manually install the compiled dependencies while using some other Python3 (such as from conda) only for the python libraries and packages.


### CMBENV

The [cmbenv tools](https://github.com/hpc4cmb/cmbenv) can generate an
install script that selectively compiles packages using specified
compilers. This allows you to \"pick and choose\" what packages are
installed from the OS versus being built from source. See the example
configs in that package and the README. For example, there is an
\"ubuntu-21.04\" config that gets everything from OS packages but also
compiles the optional dependencies like libconviqt and libmadam.

## Setting up the Python Environment



Then you can create a python3 virtualenv, activate it, and then use pip
to install these packages:

    pip install \
        numpy \
        scipy \
        matplotlib \
        healpy \
        astropy \
        pyephem \
        mpi4py


## Install Optional Dependencies



## Installing TOAST with CMake

Decide where you want to install your development copy of TOAST. I
recommend picking a standalone directory somewhere. For this example, we
will use `${HOME}/software/toast`. This should **NOT** be the same
location as your git checkout.

We want to define a small shell function that will load this directory
into our environment. You can put this function in your shell resource
file (`~/.bashrc` or `~/.profile`):

    load_toast () {
        dir="${HOME}/software/toast"
        export PATH="${dir}/bin:${PATH}"
        export CPATH="${dir}/include:${CPATH}"
        export LIBRARY_PATH="${dir}/lib:${LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${dir}/lib:${LD_LIBRARY_PATH}"
        pysite=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
        export PYTHONPATH="${dir}/lib/python${pysite}/site-packages:${PYTHONPATH}"
    }

When installing dependencies, you may have chosen to install libaatm,
libconviqt, and libmadam into this same location. If so, load this
location into your search paths now, before installing TOAST:

    load_toast

TOAST uses CMake to configure, build, and install both the compiled code
and the python tools. Within the `toast` git checkout, run the following
commands:

    mkdir -p build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=$HOME/software/toast ..
    make -j 2 install

This will compile and install TOAST in the folder `~/software/toast`.
Now, every time you want to use toast, just call the shell function:

    load_toast

If you need to customize the way TOAST gets compiled, the following
variables can be defined in the invocation to `cmake` using the `-D`
flag:

`CMAKE_INSTALL_PREFIX`

:   Location where TOAST will be installed. (We used it in the example
    above.)

`CMAKE_C_COMPILER`

:   Path to the C compiler

`CMAKE_C_FLAGS`

:   Flags to be passed to the C compiler (e.g., `-O3`)

`CMAKE_CXX_COMPILER`

:   Path to the C++ compiler

`CMAKE_CXX_FLAGS`

:   Flags to be passed to the C++ compiler

`PYTHON_EXECUTABLE`

:   Path to the Python interpreter

`BLAS_LIBRARIES`

:   Full path to the BLAS dynamical library

`LAPACK_LIBRARIES`

:   Full path to the LAPACK dynamical library

`FFTW_ROOT`

:   The install prefix of the FFTW package

`AATM_ROOT`

:   The install prefix of the libaatm package

`SUITESPARSE_INCLUDE_DIR_HINTS`

:   The include directory for SuiteSparse headers

`SUITESPARSE_LIBRARY_DIR_HINTS`

:   The directory containing SuiteSparse libraries

See the top-level \"platforms\" directory for other examples of running
CMake.

## Installing TOAST with Pip / setup.py

The setup.py that comes with TOAST is just a wrapper around the cmake
build system. You can pass options to the underlying cmake call by
setting environment variables prefixed with \"TOAST\_BUILD\_\". For
example, if you want to pass the location of the libaatm installation to
cmake when using setup.py, you can set the \"TOAST\_BUILD\_AATM\_ROOT\"
environment variable. This will get translated to \"-DAATM\_ROOT\" when
cmake is invoked by setup.py

## Warnings and Caveats

The compiled python extension in TOAST links to external libraries for BLAS/LAPACK and
FFTs.  The python Numpy package may use some of the same libraries for its internal
operations.  Inside a single process, a shared library can only be loaded once.  For
example, if numpy links to `libmkl_rt` and so does TOAST, then only one copy of
`libmkl_rt` can be loaded- and the library which actually gets loaded depends on the
order of importing the `toast` and `numpy` python modules!

For example, here are some combinations and the result:

TOAST Built With | Python Using | Result
-----------------|--------------|---------
Statically linked OpenBLAS (binary wheels on PyPI) | numpy with any LAPACK | **Works** since TOAST does not load any external LAPACK.
Intel compiler and MKL | numpy with MKL | Broken, **UNLESS** both MKLs are ABI compatible and using the Intel threading layer.
GCC compiler and MKL | numpy with MKL | Broken, since TOAST uses the MKL GNU threading layer and numpy uses the Intel threading layer (only one can be used in a single process).
GCC and system OpenBLAS | numpy with MKL | **Works**: different libraries are dl-opened by TOAST and numpy.
Intel compiler and MKL | numpy with OpenBLAS (from conda-forge) | **Works**: different libraries are dl-opened by TOAST and numpy.

When in doubt, run the toast unit tests with OpenMP enabled (i.e. `OMP_NUM_THREADS` >
1).  This should exercise code that will fail if there is some incompatibility. After
the unit tests pass, you are ready to run the benchmarks.
