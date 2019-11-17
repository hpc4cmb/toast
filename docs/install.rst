.. _install:

Installation
====================

TOAST is written in C++ and python3 and depends on several commonly available
packages.  It also has some optional functionality that is only enabled if
additional external packages are available.  The best installation method will depend on your specific needs.  We try to clarify the different options below.

User Installation
--------------------------

If you are using TOAST to build simulation and analysis workflows, including mixing built-in functionality with your own custom tools, then you can use of these methods to get started.  If you want to hack on the TOAST package itself, see the section :ref:`devinstall`.

Conda Packages
~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install TOAST and all of its optional dependencies is to use the conda package manager.  The conda-forge ecosystem allows us to create packages that are built consistently with all their dependencies.  We recommend following the `setup guidelines used by conda-forge <https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge>`_, specifically:

    1.  Install a "miniconda" base system (not the full Anaconda distribution).

    2.  Set the conda-forge channel to be the top priority package source, with strict ordering if available.

    3.  Leave the base system (a.k.a. the "root" environment) with just the bare minimum of packages.

    4.  Always create a new environment (i.e. not the base one) when setting up a python stack for a particular purpose.  This allows you to upgrade the conda base system in a reliable way, and to wipe and recreate whole conda environments whenever needed.

Here are the detailed steps of how you could do this from the UNIX shell, installing the base conda system to ``${HOME}/conda``.  First download the installer.  For OS X you would do::

    curl -SL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniconda.sh

For Linux you would do this::

    curl -SL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh

Next we will run the installer.  The install prefix should not exist previously::

    bash miniconda.sh -b -p "${HOME}/conda"

Now load this conda "root" environment::

    source ${HOME}/conda/etc/profile.d/conda.sh
    conda activate

We are going to make sure to preferentially get packages from the conda-forge channel::

    conda config --add channels conda-forge
    conda config --set channel_priority strict

Next, we are going to create a conda environment for a particular purpose (installing TOAST).  You can create as many environments as you like and install different packages within them- they are independent.  In this example, we will call this environment "toast", but you can call it anything::

    conda create -y -n toast

Now we can activate our new (and mostly empty) toast environment::

    conda activate toast

Finally, we can install the toast package.  I recommend installing the MPICH version of TOAST.  There is also a version of TOAST without MPI, but most of the parallelism in TOAST comes from using MPI::

    conda install toast=*=mpi_mpich*

OR::

    conda install toast=*=nompi*

There is also an OpenMPI version of the package, but that is mainly intended for installing toast into environments that also use / require OpenMPI.  Assuming this is the only conda installation on your system, you can add the line ``source ${HOME}/conda/etc/profile.d/conda.sh`` to your shell resource file (usually ``~/.bashrc`` on Linux or ``~/.profile`` on OS X).  You can read many articles on login shells versus non-login shells and decide where to put this line for your specific use case.

Now you can always activate your toast environment with::

    conda activate toast

And leave that environment with::

    conda deactivate

If you want to use other packages with TOAST (e.g. Jupyter Lab), then you can activate the toast environment and install them with conda.  See the conda documentation for more details on managing environments, installing packages, etc.


Minimal Install with PIP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you cannot or do not want to use the conda package manager, then it is possible to install a "minimal" version of TOAST with pip.  If you install TOAST this way, it will be missing support for MPI and atmospheric simulations.  Additionally, you must first ensure that you have a serial compiler installed and that a BLAS/LAPACK library is available in the default compiler search paths.  You should also install the FFTW package, either through your OS package manager or manually.  After doing those steps, you can do::

    $> pip install https://github.com/hpc4cmb/toast/archive/2.3.3.tar.gz

Specify the URL to the version tarball you want to install (see the releases on the TOAST github page).


Something Else
~~~~~~~~~~~~~~~~~~~~~

If you have a custom install situation that is not met by the above solutions, then you should follow the instructions below for a "Developer install".


.. _devinstall:

Developer Installation
-----------------------------

Before setting up a software stack for TOAST development, you should become familiar with the following:

    1.  What serial C++11 compilers are you using?

    2.  If using MPI, your MPI installation must be compatible with your serial compilers.

    3.  What python3 installation are you using?

    4.  Your mpi4py installation must be compatible with #3 and #2.

The rest of this section describes a recommended procedure for getting everything set up.  If you are an expert in software package management then there are many variations that will work, but there are also many pitfalls.

Compilers
~~~~~~~~~~~~~~~~

On Linux, you are likely using the GNU compilers.  You can install these through your OS package manager.  On OS X, you can use the clang compiler provided by XCode developer tools, or the macports and homebrew packaging systems provide versions of newer GNU compilers.

When choosing an MPI version, we recommend installing an MPICH version from the same source as your serial compilers (Linux package manager, homebrew, macports, etc).  You can also use OpenMPI, but make sure that you can run an MPI program before trying to use it with TOAST.  Some Linux installs of OpenMPI require you to edit a config file for your system first.  If there is no compatible MPI version available, you can install MPICH manually using your serial compilers, or just skip using MPI on your development system.

There also exist conda packages for compilers, but I recommend *not* using those for compiling general software outside of conda packages.  The reason is that these compilers are cross-compilers and care needs to be taken to ensure that they are not mixed with other system libraries.


Python 3
~~~~~~~~~~~~~

You can obtain a compatible version of Python (>=3.4) from OS packages, homebrew / macports, or conda.  In addition to Python, you will need to install::

    numpy
    scipy
    astropy
    healpy
    pyephem

If you are using an OS packaged python, you likely want to first create a virtualenv before installing these packages with pip.  If you are using conda, consider creating a new env for this development work.


Compiled Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

After selecting your compilers and your Python, choose a separate installation prefix to install TOAST and any manually compiled dependencies.  Do not install this software into a conda environment.  For this example, we will put our compiled dependencies and TOAST into ``${HOME}/toastdev``.

The required dependencies for TOAST are LAPACK/BLAS, FFTW, and CMake.  If you want to do atmosphere simulations, then you additionally need to install SuiteSparse and the libaatm package.  On a recent Ubuntu, you can get all of these with::

    apt update
    apt install \
        cmake \
        libopenblas-dev \
        liblapack-dev \
        libfftw3-dev \
        libsuitesparse-dev

To install libaatm, `download a release tarball here <https://github.com/hpc4cmb/libaatm/releases>`_


Compiled Dependencies
--------------------------

TOAST compilation requires a C++11 compatible compiler as well as a compatible
MPI C++ compiler wrapper.  You must also have an FFT library and both FFTW and
Intel's MKL are supported by configure checks.  Additionally a BLAS/LAPACK
installation is required.

Several optional compiled dependencies will enable extra features in
TOAST.  If the `Elemental library <http://libelemental.org/>`_ is
found at configure time then internal atmosphere simulation code will
be enabled in the build.  If the `MADAM destriping mapmaker
<https://github.com/hpc4cmb/libmadam>`_ is available at runtime, then
the python code will support calling that library; the same applies
for `Libconviqt <https://github.com/hpc4cmb/libconviqt>`_.

You are advised to setup and install these dependencies in a dedicated
folder, like ``$HOME/toast-deps``. For ``libmadam`` and
``libconviqt``, you should compile both of them as follows::

    mkdir $HOME/toast-deps
    ./autogen.sh && ./configure --prefix=$HOME/toast-deps && make && make install

Then you should updated ``LD_LIBRARY_PATH`` to point to
``$HOME/toast-deps/lib``: we will show how to do this below.


Python Dependencies
------------------------

You should have a reasonably new (>= 3.4.0) version of python3.  We also require
several common scientific python packages:

    * numpy
    * scipy
    * matplotlib
    * pyephem
    * mpi4py (>= 2.0.0)
    * healpy
    * sphinx, sphinx_rtd_theme (if you want to rebuild the documentation)

For mpi4py, ensure that this package is compatible with the MPI C++ compiler
used during TOAST installation.  When installing healpy, you might encounter
difficulties if you are in a cross-compile situation.  In that case, I
recommend installing the `repackaged healpix here <https://github.com/tskisner/healpix-autotools>`_.

There are obviously several ways to meet these python requirements.

Option #0
~~~~~~~~~~~~~

If you are using machines at NERSC, see :ref:`nersc`.

Option #1
~~~~~~~~~~~~~

If you are using a linux distribution which is fairly recent (e.g. the
latest Ubuntu version), then you can install all the dependencies with
the system package manager::

    %> apt-get install fftw-dev python3-scipy \
       python3-matplotlib python3-ephem python3-healpy \
       python3-mpi4py

On OS X, you can also get the dependencies with macports.  However, on some
systems OpenMPI from macports is broken and MPICH should be installed
as the dependency for the mpi4py package.

Option #2
~~~~~~~~~~~~~

If your OS is old, you could use a virtualenv to install updated versions
of packages into an isolated location.  This is also useful if you want to
separate your packages from the system installed versions, or if you do not
have root access to the machine.  Make sure that you have python3 and the
corresponding python3-virtualenv packages installed on your system.  Also
make sure that you have some kind of MPI (OpenMPI or MPICH) installed with
your system package manager.  Then:

    1.  create a virtualenv and activate it.

    2.  once inside the virtualenv, pip install the dependencies

Option #3
~~~~~~~~~~~~~~

Use Anaconda.  Download and install Miniconda or the full Anaconda distribution.
Make sure to install the Python3 version.  If you are starting from Miniconda,
install the dependencies that are available through conda::

    %> conda install -c conda-forge numpy scipy matplotlib mpi4py healpy pyephem

Using CMake
-----------------------

TOAST uses CMake to configure, build, and install both the compiled code
and the python tools.  The suggested way to run TOAST is to install it in a
directory within your home folder, and then make it visible using the
``PYTHONPATH`` environment variable.

Within the ``toast`` directory, run the following commands::

    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=$HOME/toast ..    % Pick the directory you want
    make && make install

This will compile and install TOAST in the folder ``~/toast``. Now, every
time you want to run TOAST you must tell Python where it was installed::

    export PATH=$HOME/toast/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/toast-deps:$LD_LIBRARY_PATH
    export PYTHONPATH=$HOME/toast/lib/python${PYVER}/site-packages/:$PYTHONPATH

Replace ``${PYVER}`` with the version of the Python interpreter you
are using (e.g., ``3.7``). This should be enough to run TOAST. (If you
do not like to run the previous line every time you need TOAST, you
can put it in your ``.profile``.)

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

    cd docs && make html

The documentation will be available in ``docs/_build/html``.
