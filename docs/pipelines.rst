.. _pipelines:

Pipelines
=================================

TOAST workflows are usually called "pipelines" and consist of a :class:`toast.Data` object that is passed through one or more "operators":

.. autoclass:: toast.Operator
    :members:

There are very few restrictions on an "operator" class.  It can have arbitrary constructor arguments and must define an `exec()` method which takes a `toast.Data` instance.

Each operator might take many arguments.  There are helper functions in `toast.pipeline_tools` that can be used to create an operator in a pipeline.  Currently these helper functions add arguments to `argparse` for control at the command line.  In the future, we intend to support loading operator configuration from other config file formats.


Example:  Simple Satellite Simulation
-----------------------------------------

TOAST includes several "generic" pipelines that simulate some fake data and then run some operators on that data.  One of these is installed as `toast_satellite_sim.py`.  There is some "set up" in the top of the script, but if we remove the timing code then the `main()` looks like this:

.. include:: pipe_satellite_sim.inc
