.. _install:

Installation
====================

PyTOAST is written in python3 and depends on several commonly available
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
    * quaternionarray
    * matplotlib
    * mpi4py (>= 2.0.0)
    * cython (if installed from a git checkout)
    

All of these are available from standard locations (e.g. PyPi, with
"pip install <package>").  Make sure to install the python3 versions of
these!

In order to not mess up your system-installed python tools, I highly
recommend this procedure:

    1. install python3 and python3-virtualenv with your standard package
       manager (apt, macports, cygwin, etc).

    2. create a virtualenv for your work.

    3. inside the virtualenv, pip install the dependencies above.


Using setup.py
-----------------------

You can install a stand-alone version of the package like this::

    %> python3 setup.py install --prefix=/some/path

If you don't specify the prefix, it will install to your main virtualenv
location, or to the main python install location on the system.  This
might not be what you want.  Installing in "development" mode is somewhat
more complex, due to the use of compiled C libraries and extensions.  See
more details in the :ref:`dev`.

After installation, you can run a set of unit tests with::

    %> python3 setup.py test

