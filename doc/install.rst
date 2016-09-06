.. _install:

Installation
====================

TOAST is written in python3 and depends on several commonly available
packages.  It also has some optional functionality for calling externally
installed packages.


Dependencies
-------------------

This package uses python 3, which was first released in 2008.  The setup.py
file checks for python >= 3.4.0, but it may work with older versions of
python 3.

Other dependencies are:

    * numpy
    * scipy
    * healpy
    * matplotlib
    * mpi4py (>= 2.0.0)
    * cython (if installed from a git checkout)

Option #0
~~~~~~~~~~~~~

If you are using NERSC, see :ref:`nersc`.

Option #1
~~~~~~~~~~~~~

If you are using a linux distribution which is fairly recent (e.g. the
latest Ubuntu version), then you can install all the dependencies with
the system package manager::

    %> apt-get install python3-scipy \
       python3-matplotlib python3-healpy \
       cython3 python3-mpi4py

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

    %> conda install numpy scipy matplotlib cython mpi4py

Then install healpy with pip::

    %> pip install healpy


Using setup.py
-----------------------

You can install a stand-alone version of the toast package like this::

    %> python3 setup.py install --prefix=/some/path

If you don't specify the prefix, it will install to your main python location,
which will depend on how you installed the dependencies above.  

Please give some thought to your development workflow.  If you are using a system
installed python, don't install toast to that location.  Create a new directory
somewhere and add that location prefix to PATH and PYTHONPATH before installing.
If you are using a virtualenv or an Anaconda environment, it might be fine to 
just install to that location.

Installing in "development" mode is somewhat more complex, due to the use of 
compiled C libraries and extensions.  See more details in the :ref:`dev`.

