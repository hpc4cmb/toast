.. _utilsenv:

Environment Control
--------------------------------

The run-time behavior of the TOAST package can be controlled by the manipulation of several environment variables.   The current configuration can also be queried.

.. autoclass:: toast.utils.Environment
    :members:


Logging
--------------------------------

Although python provides logging facilities, those are not accessible to C++.  The logging class provided in TOAST is usable from within the compiled libtoast code and also from python, and uses logging level independent from the builtin python logger.

.. autoclass:: toast.utils.Logger
    :members:
