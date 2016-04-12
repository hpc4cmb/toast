.. _install:

Overview
=================================

PyTOAST is written in python3 and depends on several commonly available
packages.  It also has some optional functionality for calling externally
installed packages.


Dependencies
=================================

This package uses python 3, which was first released in 2008.  The setup.py
file checks for python >= 3.4.0, but it may work with older versions of
python 3.

Other dependencies are:

    * numpy
    * scipy
    * healpy
    * quaternionarray
    * matplotlib
    * cython
    * mpi4py (>= 2.0.0)

All of these are available from standard locations (e.g. PyPi, with
"pip install <package>").  Make sure to install the python3 versions of
these!

In order to not mess up your system-installed python tools, I highly
recommend this procedure:

    1. install python3 and python3-virtualenv with your standard package
       manager (apt, macports, cygwin, etc).

    2. create a virtualenv for your work.

    3. inside the virtualenv, pip install the dependencies above.


Installation
=================================

There are 2 choices when installing pytoast.  You can install a
stand-alone version of the package like this::

    %> python3 setup.py install --prefix=/some/path

If you don't specify the prefix, it will install to your main virtualenv
location, or to the main python install location on the system.  This
might not be what you want.  You can also install a "link" that points
back to your active git source tree and uses whatever is there.  This is
useful for development::

    %> python3 setup.py develop --prefix=/some/path

Note that if new scripts or modules are added to the git checkout, you
will have to re-run the command above to pick up those changes.  Also,
if you change the cython sources, you should re-run the develop command.
