.. _dev:

Developer's Guide
====================

PyTOAST aims to follow Python best practices whenever reasonably possible.  If you are submitting a pull request to contribute some code, please ensure that you are using indents consisting of 4 spaces (no tabs).  Also look at `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_.  Contributions correcting existing code are also welcome!

When documenting classes and methods, we use `google-style docstrings <http://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`_.


Running from the Source Tree
--------------------------------

For rapid development, it is sometimes useful to work directly in the source tree.  The standard setup.py "develop" target does not work well for pytoast, since it contains a compiled C library as well as compiled extensions.  Here are the recommended steps::

    %> ./develop clean
    %> ./develop build
    %> ./develop test <test_file.py>
    %> ./develop env


