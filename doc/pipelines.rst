.. _pipelines:

Pipelines
=================================

Before delving into the structure of the toast package, it is sometimes
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
requires the use of libmadam (the ``--madam`` option).

.. include:: pipe_satellite_sim.inc


Example:  Proposed CoRE Satellite Boresight
----------------------------------------------

Here is one example using this script to generate one day of scanning with a single boresight detector, and using one proposed scan strategy for a LiteCoRE satellite::

    toast_satellite_sim.py --samplerate 175.86 --spinperiod 1.0 --spinangle 45.0 
    --precperiod 5760.0 --precangle 50.0 --hwprpm 0.0 --obs 23.0 --gap 1.0 
    --obschunks 24 --numobs 1 --nside 1024 --baseline 5.0 --madam --noisefilter 
    --fp fp_core.pkl --outdir out_core_nohwp_fast


Example:  Proposed LiteBIRD Satellite Boresight
------------------------------------------------------

Here is how you could do a similar thing with a boresight detector and one proposed lightbird scanning strategy for a day::

    toast_satellite_sim.py --samplerate 23.0 --spinperiod 10.0 --spinangle 30.0 
    --precperiod 93.0 --precangle 65.0 --hwprpm 88.0 --obs 23.0 --gap 1.0 
    --obschunks 24 --numobs 1 --nside 1024 --baseline 60.0 --madam --fp fp_lb.pkl
    --debug --outdir out_lb_hwp


Creating Your Own Pipeline
------------------------------

TOAST is designed to give you tools to piece together your own data processing workflow.  Here is a slightly modified version of the pipeline script above.  This takes a boresight detector with 1/f noise properties, simulates a sort-of Planck scanning strategy, generates a noise timestream, and then generates a fake signal timestream and adds it to the noise.  Then it uses madam to make a map.

.. literalinclude:: ../examples/toast_example_customize.py

