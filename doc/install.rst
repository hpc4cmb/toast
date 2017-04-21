.. _install:

Installation
====================

TOAST is written in C++ and python3 and depends on several commonly available
packages.  It also has some optional functionality that is only enabled if 
additional external libraries are available.


Compiled Dependencies
--------------------------

TOAST compilation requires a C++11 compatible compiler as well as a compatible
MPI C++ compiler wrapper.  You must also have an FFT library and both FFTW and 
Intel's MKL are supported by configure checks.  Additionally a BLAS/LAPACK 
installation is required.

Several optional compiled dependencies will enable extra features in TOAST.
If the `Elemental library <http://libelemental.org/>`_ is found at configure 
time then internal atmosphere simulation code will be enabled in the build.
If the `MADAM destriping mapmaker <https://github.com/hpc4cmb/libmadam>`_ is
available at runtime, then the python code will support calling that library.


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

    %> conda install numpy scipy matplotlib mpi4py

Then install healpy and pyephem with pip::

    %> pip install healpy pyephem


Using Configure
-----------------------

TOAST uses autotools to configure, build, and install both the compiled code
and the python tools.  If you are running from a git checkout (instead of a
distribution tarball), then first do::

    %> ./autogen.sh

Now run configure::

    %> ./configure --prefix=/path/to/install

See the top-level "platforms" directory for other examples of running the
configure script.  Now build and install the tools::

    %> make install

In order to use the installed tools, you must make sure that the installed
location has been added to the search paths for your shell.  For example,
the "<prefix>/bin" directory should be in your PATH and the python install
location "<prefix>/lib/pythonX.X/site-packages" should be in your PYTHONPATH.


Testing the Installation
-----------------------------

After installation, you can run both the compiled and python unit tests.
These tests will create an output directory in your current working directory::

    %> python -c "import toast; toast.test()"


