# Introduction

The Time-Ordered Astrophysics Scalable Tools (TOAST) package is a [software framework](https://en.wikipedia.org/wiki/Software_framework)
designed for simulation and reduction of data from telescope receivers which acquire
timestreams of individual detector responses.  This type of instrumentation is often
used when observing wavelengths from the microwave through the far IR.  Data from such
telescopes present unique challenges compared to images acquired in optical astronomy.
Timestreams of detector responses can be correlated in time due to the optical response
time of the detector and noise in the readout chain.  Timestreams can be correlated
between detectors due to thermal fluctuations, readout cross-talk, atmospheric emission
and pickup from the ground.  These correlations motivate us to process larger pieces of
data simultaneously in order to extract the best possible estimate of the underlying
signal coming from space, while reducing statistical errors and removing sources of
systematic error.  To accommodate the collective processing of large data sets, TOAST
has been designed to scale to high concurrency to handle the largest data volumes.
TOAST workflows have been run on systems ranging from a laptop up to 150,000 cores of
the largest HPC system at the NERSC computing center.  The software has been developed
over the course of 15 years and three major revisions.

The TOAST framework allows the user to build up a simulation and reduction workflow
composed of well tested and general tools, as well as custom operations specific to a
given experiment.  TOAST provides modules for simulating sky signal, correlated
instrument noise, readout crosstalk, realistic atmospheric signal, beam asymmetries,
ground pickup, calibration errors, pointing errors, and more.  For data reduction, the
package provides modules for a variety of time domain and spatial filters, simple binned
mapmaking, and a "generalized destriping" mapmaker which can regress templates from the
data to model the noise, gain drifts, and other contaminating signals.  Along with these
built-in modules, experiments can create their own simulation and reduction modules to
build a workflow customized to a specific use case.

The TOAST framework has been used in previous experiments, including the
[joint LFI/HFI Planck analysis](https://arxiv.org/abs/2007.04997).  The TOAST framework
is one component of the data management tools being developed by the Simons Observatory.
TOAST is also part of the baseline plan for CMB-S4 simulation and reduction at HPC
centers such as NERSC and ALCF.

## Specific Experiments

If you are working with data from or simulations of one of these projects:

- Planck
- LiteBIRD
- Simons Observatory
- CMB-S4
- TolTEC

Then there are additional software repositories you have access to that contain extra
TOAST classes and scripts for your experiment.
