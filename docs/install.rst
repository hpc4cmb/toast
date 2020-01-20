.. _install:

Installation
====================

TOAST is written in C++ and python3 and depends on several commonly available packages.
It also has some optional functionality that is only enabled if additional external
packages are available.  The best installation method will depend on your specific
needs.  We try to clarify the different options below.

User Installation
--------------------------

If you are using TOAST to build simulation and analysis workflows, including mixing
built-in functionality with your own custom tools, then you can use one of these methods
to get started.  If you want to hack on the TOAST package itself, see the section
:ref:`devinstall`.

If you want to use TOAST at NERSC, see :ref:`nersc`.

Conda Packages
~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install TOAST and all of its optional dependencies is to use the
conda package manager.  The conda-forge ecosystem allows us to create packages that are
built consistently with all their dependencies.  If you already have Anaconda installed
**and** it is a recent version, then you may be able to skip ahead to "activating the
root environment below".

We recommend following the `setup guidelines used by conda-forge
<https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge>`_,
specifically:

    1.  Install a "miniconda" base system (not the full Anaconda distribution).

    2.  Set the conda-forge channel to be the top priority package source, with strict ordering if available.

    3.  Leave the base system (a.k.a. the "root" environment) with just the bare minimum of packages.

    4.  Always create a new environment (i.e. not the base one) when setting up a python stack for a particular purpose.  This allows you to upgrade the conda base system in a reliable way, and to wipe and recreate whole conda environments whenever needed.

Here are the detailed steps of how you could do this from the UNIX shell, installing the
base conda system to ``${HOME}/conda``.  First download the installer.  For OS X you
would do::

    curl -SL \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh \
    -o miniconda.sh

For Linux you would do this::

    curl -SL \
    https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o miniconda.sh

Next we will run the installer.  The install prefix should not exist previously::

    bash miniconda.sh -b -p "${HOME}/conda"

Now load this conda "root" environment::

    source ${HOME}/conda/etc/profile.d/conda.sh
    conda activate

We are going to make sure to preferentially get packages from the conda-forge channel::

    conda config --add channels conda-forge
    conda config --set channel_priority strict

Next, we are going to create a conda environment for a particular purpose (installing
TOAST).  You can create as many environments as you like and install different packages
within them- they are independent.  In this example, we will call this environment
"toast", but you can call it anything::

    conda create -y -n toast

Now we can activate our new (and mostly empty) toast environment::

    conda activate toast

Finally, we can install the toast package.  I recommend installing the MPICH version of
TOAST.  There is also a version of TOAST without MPI, but most of the parallelism in
TOAST comes from using MPI::

    conda install python=3 toast=*=*mpich*

OR::

    conda install python=3 toast=*=*nompi*

There is also an OpenMPI version of the package, but that is mainly intended for
installing toast into environments that also use / require OpenMPI.  Assuming this is
the only conda installation on your system, you can add the line ``source
${HOME}/conda/etc/profile.d/conda.sh`` to your shell resource file (usually
``~/.bashrc`` on Linux or ``~/.profile`` on OS X).  You can read many articles on login
shells versus non-login shells and decide where to put this line for your specific use
case.

Now you can always activate your toast environment with::

    conda activate toast

And leave that environment with::

    conda deactivate

If you want to use other packages with TOAST (e.g. Jupyter Lab), then you can activate
the toast environment and install them with conda.  See the conda documentation for more
details on managing environments, installing packages, etc.

If you want to use PySM with TOAST for sky simulations, you should install the ``pysm3``
and ``libsharp`` packages.  The libsharp package has the same variants as the toast
package, so you should choose the same variant as you used when installing TOAST.  For
example::

    conda install pysm3 libsharp=*=*mpich*

OR::

    conda install pysm3 libsharp=*=*nompi*


Minimal Install with PIP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you cannot or do not want to use the conda package manager, then it is possible to
install a "minimal" version of TOAST with pip.  If you install TOAST this way, it will
be missing support for MPI and atmospheric simulations.  Additionally, you must first
ensure that you have a serial compiler installed and that a BLAS/LAPACK library is
available in the default compiler search paths.  You should also install the FFTW
package, either through your OS package manager or manually.  After doing those steps,
you can do::

    pip install https://github.com/hpc4cmb/toast/archive/2.3.5.tar.gz

Specify the URL to the version tarball you want to install (see the releases on the
TOAST github page).


Something Else
~~~~~~~~~~~~~~~~~~~~~

If you have a custom install situation that is not met by the above solutions, then you
should follow the instructions below for a "Developer install".


.. _devinstall:

Developer Installation
-----------------------------

Here we will discuss several specific system configurations that are known to work.  The
best one for you will depend on your OS and preferences.

Ubuntu Linux
~~~~~~~~~~~~~~~~

You can install all but one required TOAST dependency using packages provided by the OS.
Note that this assumes a recent version of ubuntu (tested on 19.04)::

    apt update
    apt install \
        cmake \
        build-essential \
        gfortran \
        libopenblas-dev \
        libmpich-dev \
        liblapack-dev \
        libfftw3-dev \
        libsuitesparse-dev \
        python3-dev \
        libpython3-dev \
        python3-scipy \
        python3-matplotlib \
        python3-healpy \
        python3-astropy \
        python3-pyephem


NOTE:  if you are using another package on your system that requires OpenMPI, then you
may get a conflict installing libmpich-dev.  In that case, just install libopenmpi-dev
instead.

Next, download a `release of libaatm <https://github.com/hpc4cmb/libaatm/releases>`_ and
install it.  For example::

    cd libaatm
    mkdir build
    cd build
    cmake \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        ..
    make -j 4
    sudo make install

You can also install it to the same prefix as TOAST or to a separate location for just
the TOAST dependencies.  If you install it somewhere other than /usr/local then make
sure it is in your environment search paths (see the "installing TOAST" section).

You can also now install the optional dependencies:

    * `libconviqt <https://github.com/hpc4cmb/libconviqt>`_ for 4PI beam convolution.
    * `libmadam <https://github.com/hpc4cmb/libmadam>`_ for optimized destriping mapmaking.


Other Linux
~~~~~~~~~~~~~~~~

If you have a different distro or an older version of Ubuntu, you should try to install
at least these packages with your OS package manager::

    gcc
    g++
    mpich or openmpi
    lapack
    fftw
    suitesparse
    python3
    python3 development library (e.g. libpython3-dev)
    virtualenv (e.g. python3-virtualenv)

Then you can create a python3 virtualenv, activate it, and then use pip to install these
packages::

    pip install \
        scipy \
        matplotlib \
        healpy \
        astropy \
        pyephem

Then install libaatm as discussed in the previous section.


Conda Isolated Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is still a work in progress.  Conda provides compilers as well as packages, but in
order to use them we must isolate **everything** from the surrounding OS.  The obvious
appeal is that we can then install all dependencies easily and just build TOAST using
the conda compilers.  We will add more details here after more testing.

OS X with MacPorts
~~~~~~~~~~~~~~~~~~~~~~

.. todo::  Document using macports to get gcc and installing optional dependencies.

OS X with Homebrew
~~~~~~~~~~~~~~~~~~~~~~~~

.. todo::  Document installing compiled dependencies and using a virtualenv.

Full Custom Install with CMBENV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `cmbenv package <https://github.com/hpc4cmb/cmbenv>`_ can generate an install script
that selectively compiles packages using specified compilers.  This allows you to "pick
and choose" what packages are installed from the OS versus being built from source.  See
the example configs in that package and the README.  For example, there is an
"ubuntu-19.04" config that gets everything from OS packages but also compiles the
optional dependencies like libconviqt and libmadam.


Installing TOAST
~~~~~~~~~~~~~~~~~~~~~~~~

Decide where you want to install your development copy of TOAST.  I recommend picking a
standalone directory somewhere.  For this example, we will use
```${HOME}/software/toast``.  This should **NOT** be the same location as your git
checkout.

We want to define a small shell function that will load this directory into our
environment.  You can put this function in your shell resource file (``~/.bashrc`` or
``~/.profile``)::

    load_toast () {
        dir="${HOME}/software/toast"
        export PATH="${dir}/bin:${PATH}"
        export CPATH="${dir}/include:${CPATH}"
        export LIBRARY_PATH="${dir}/lib:${LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${dir}/lib:${LD_LIBRARY_PATH}"
        pysite=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
        export PYTHONPATH="${dir}/lib/python${pysite}/site-packages:${PYTHONPATH}"
    }

When installing dependencies, you may have chosen to install libaatm, libconviqt, and
libmadam into this same location.  If so, load this location into your search paths now,
before installing TOAST::

    load_toast

TOAST uses CMake to configure, build, and install both the compiled code
and the python tools.  Within the ``toast`` git checkout, run the following commands::

    mkdir -p build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=$HOME/software/toast ..
    make -j 2 install

This will compile and install TOAST in the folder ``~/software/toast``. Now, every
time you want to use toast, just call the shell function::

    load_toast

If you need to customize the way TOAST gets compiled, the following
variables can be defined in the invocation to ``cmake`` using the
``-D`` flag:

``CMAKE_INSTALL_PREFIX``
    Location where TOAST will be installed. (We used it in the example above.)

``CMAKE_C_COMPILER``
    Path to the C compiler

``CMAKE_C_FLAGS``
    Flags to be passed to the C compiler (e.g., ``-O3``)

``CMAKE_CXX_COMPILER``
    Path to the C++ compiler

``CMAKE_CXX_FLAGS``
    Flags to be passed to the C++ compiler

``MPI_C_COMPILER``
    Path to the MPI wrapper for the C compiler

``MPI_CXX_COMPILER``
    Path to the MPI wrapper for the C++ compiler

``PYTHON_EXECUTABLE``
    Path to the Python interpreter

``BLAS_LIBRARIES``
    Full path to the BLAS dynamical library

``LAPACK_LIBRARIES``
    Full path to the LAPACK dynamical library

``FFTW_ROOT``
    The install prefix of the FFTW package

``SUITESPARSE_INCLUDE_DIR_HINTS``
    The include directory for SuiteSparse headers

``SUITESPARSE_LIBRARY_DIR_HINTS``
    The directory containing SuiteSparse libraries

See the top-level "platforms" directory for other examples of running CMake.


Testing the Installation
-----------------------------

After installation, you can run both the compiled and python unit
tests.  These tests will create an output directory named ``out`` in
your current working directory::

    python -c "import toast.tests; toast.tests.run()"


Building the Documentation
-----------------------------

You will need the two Python packages ``sphinx`` and
``sphinx_rtd_theme``, which can be installed using ``pip`` or
``conda`` (if you are running Anaconda)::

    cd docs && make clean && make html

The documentation will be available in ``docs/_build/html``.
