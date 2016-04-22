.. _pipelines:

Pipelines
=================================

Before delving into the structure of the pytoast package, it is sometimes
useful to look at (and use!) an example.  One such program is the simple script
below which simulates a fake satellite scanning strategy with a focalplane of
detectors and then makes a map.


Simple Satellite Simulation
-----------------------------------

The current version of this tool simulates parameterized boresight pointing
and then uses the given focalplane (loaded from a pickle file) to compute
the detector pointing.  Noise properties of each detector are used to
simulate noise timestreams.

In order to create a focalplane file, you can do for example::

  import pickle
  import numpy as np

  fake = {}
  fake['quat'] = np.array([0.0, 0.0, 1.0, 0.0])
  fake['fwhm'] = 30.0
  fake['fknee'] = 0.05
  fake['alpha'] = 1.0
  fake['NET'] = 0.000060
  fake['color'] = 'r'
  fp = {}
  fp['bore'] = fake

  with open('fp_lb.pkl', 'wb') as p:
      pickle.dump(fp, p)

Note that until the older TOAST mapmaking tools are ported, this script
requires the use of libmadam (the --madam option).

.. include:: pipe_satellite_sim.inc


Example:  Proposed CoRE Satellite Boresight
----------------------------------------------

TODO


Example:  Proposed LiteBIRD Satellite Boresight
------------------------------------------------------

TODO


Creating Your Own Pipeline
------------------------------

PyTOAST is designed to give you tools to piece together your own data processing workflow.

