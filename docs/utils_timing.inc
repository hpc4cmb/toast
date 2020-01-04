.. _utilstiming:

Timing
-----------------------------------

TOAST includes some utilities for setting arbitrary timers on individual processes and also gathering timer data across MPI processes.  A basic manual timer is implemented in the ``Timer`` class:

.. autoclass:: toast.timing.Timer
    :members:

A single ``Timer`` instance is only valid until it goes out of scope.  To start and stop a global timer that is accessible anywhere in the code, use the ``get()`` method of the ``GlobalTimers`` singleton to get a handle to the single instance and then use the class methods to start and stop a named timer:

.. autoclass:: toast.timing.GlobalTimers
    :members:

The ``GlobalTimers`` object can be used to automatically time all calls to a particular function by using the ``@function_timer`` decorator when defining the function.

To gather timing statistics across all processes you can use this function:

.. autofunction:: toast.timing.gather_timers

And the resulting information can be written on the root process with:

.. autofunction:: toast.timing.dump
