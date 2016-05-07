.. _dev:

Developer's Guide
====================

TOAST aims to follow Python best practices whenever reasonably possible.  If you are submitting a pull request to contribute some code, please ensure that you are using indents consisting of 4 spaces (no tabs).  Also look at `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_.  Contributions correcting existing code are also welcome!

When documenting classes and methods, we use `google-style docstrings <http://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`_.


Running from the Source Tree
--------------------------------

For rapid development, it is sometimes useful to work directly in the source tree.  The standard setup.py "develop" target does not work well for pytoast, since it contains a compiled C library as well as compiled extensions.  Here are the recommended steps.  Instead, you can use the enclosed top-level script called "develop".  To clean all build products, do::

    %> ./develop clean

To build everything inside the source tree, do::

    %> ./develop build

To run a single file in the unittest directory, you can do::

    %> ./develop test <test_file.py>

Or, if you leave out the name of the test file then all tests are run.  If you want to prepend your source tree to your PATH and PYTHONPATH, you can use this command to get the necessary shell modifiers::

    %> ./develop env

To actually run those commands, do this::

    %> eval `./develop env`

Now your source tree is in your search paths, and you can go to other directories and import the toast module and run the pipeline scripts.

