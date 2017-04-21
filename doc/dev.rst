.. _dev:

Developer's Guide
====================

TOAST aims to follow best practices whenever reasonably possible.  If you submit a pull request to contribute C++ code, try to match the existing coding style (indents are 4 spaces, not tabs, curly brace placement, spacing, etc).  If you are contributing python code follow `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_.  When documenting python classes and methods, we use `google-style docstrings <http://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`_.  The C++ code in TOAST uses the google test framework for unit tests.  Python code uses the standard built in unittest classes.  When contributing new code, please add unit tests as well.  Even if we don't have perfect test coverage, that should be our goal.  When actively developing the codebase, you can run the C++ unit tests without installation by doing::

    %> make check

In order to run the python unit tests, you must first do a "make install".

