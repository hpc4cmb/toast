.. _pipelines:

Pipelines
=================================

TOAST pipelines are a top-level python script which instantiates a `toast.Data` object
as well as one or more `toast.Operator` classes which are applied to the data.

Each operator might take many arguments in its constructor.  There are helper functions
in `toast.pipeline_tools` that can be used to create some built-in operators in a
pipeline script.  Currently these helper functions add arguments to an `argparse`
namespace for control at the command line.  In the future, we intend to support loading
operator configuration from other config file formats.

The details of how the global data object is created will depend on a particular project
and likely use classes specific to that experiment.  Here we look at several examples
using built-in classes.


Example:  Simple Satellite Simulation
-----------------------------------------

TOAST includes several "generic" pipelines that simulate some fake data and then run some operators on that data.  One of these is installed as `toast_satellite_sim.py`.  There is some "set up" in the top of the script, but if we remove the timing code then the `main()` looks like this:

.. include:: pipe_satellite_sim.inc
