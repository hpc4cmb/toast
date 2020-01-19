.. _operators:

Operators
=================

TOAST workflows ("pipelines") consist of a :class:`toast.Data` object that is passed through one or more "operators":

.. autoclass:: toast.Operator
    :members:

There are very few restrictions on an "operator" class.  It can have arbitrary
constructor arguments and must define an `exec()` method which takes a `toast.Data`
instance.  TOAST ships with many built-in operators that are detailed in the rest of
this section.  Operator constructors frequently require many options.  Most built-in
operators have helper functions in the `toast.pipeline_tools` module to ease parsing of
these options from the command line or argparse input file.

.. todo:: Document the pipeline_tools functions for each built-in operator.

.. include:: op_pointing.inc

.. include:: op_sim_signal.inc

.. include:: op_sim_noise.inc

.. include:: op_processing.inc

.. include:: op_mapmaking.inc
