.. _dev:

Developer's Guide
====================

Notes to add:  C++ class / function naming conventions, uncrustify, pep8 linter, etc.

TOAST aims to follow best practices whenever reasonably possible.  If you submit a pull request to contribute C++ code, try to match the existing coding style (indents are 4 spaces, not tabs, curly brace placement, spacing, etc).  If you are contributing python code follow `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_.  When documenting python classes and methods, we use `google-style docstrings <http://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`_.  The C++ code in TOAST uses the google test framework for unit tests.  Python code uses the standard built in unittest classes.  When contributing new code, please add unit tests as well.  Even if we don't have perfect test coverage, that should be our goal.  When actively developing the codebase, you can run the C++ unit tests without installation by doing::

    %> make check

In order to run the python unit tests, you must first do a "make install".



Compiled Code
-------------------

Class names follow python CamelCase convention
Function names follow python_underscore_convention

Formatting set by uncrustify.

All code that is exposed through pybind11 in a single toast namespace.  Nested namespaces may be used for code that is internal to the C++ code.

The "using" statement can be used for aliasing a specific class or type::

    using ShapeContainer = py::detail::any_container<ssize_t>;

But should **not** be used to import an entire namespace::

    using std;

Pointer / reference declaration.  This allows reading from right to left as "a pointer to a constant double" or "a reference to a constant double".

    double const * data
    double const & data

Not:

    const double * data
    const double & data

When indexing the size of an STL container, the index variable should be either of the size type declared in the container class or size_t.

When describing time domain sample indices or intervals, we using int64_t everywhere for consistency.  This allows passing, e.g. "-1" to communicate unspecified intervals or sample indices.

Single line conditional statements:

    if (x > 0) y = x;

Are permitted if they fit onto a single line.  Otherwise, insert braces.

Internal toast source files should not include the main "toast.hpp".  Instead
they should include the specific headers they need.  For example::

    #include <toast/sys_utils.hpp>
    #include <toast/math_lapack.hpp>
    #include <toast/math_qarray.hpp>


If attempting to vectorize code with OpenMP simd constructs, be sure to check that any data array used in the simd region are aligned (see toast::is_aligned).  Otherwise this can result in silent data corruption.
