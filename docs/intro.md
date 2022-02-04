
# Introduction

TOAST is a [software framework](https://en.wikipedia.org/wiki/Software_framework) for
simulating and processing timestream data collected by telescopes. Telescopes which
collect data as timestreams rather than images give us a unique set of analysis
challenges. Detector data usually contains noise which is correlated in time as well as
sources of correlated signal from the instrument and the environment. Large pieces of
data must often be analyzed simultaneously to extract an estimate of the sky signal.
TOAST has evolved over several years. The current codebase contains an internal C++
library to allow for optimization of some calculations, while the public interface is
written in Python.

The TOAST framework contains:

- Tools for distributing data among many processes
- Tools for performing operations on the local pieces of the data
- Generic operators for common processing tasks (filtering, pointing expansion, map-making)
- Basic classes for performing I/O in a limited set of formats
- Well-defined interfaces for adding custom I/O classes and processing operators

The highest-level control of the workflow is done by the user, often by writing a small
Python script or notebook (some examples are included).  Such scripts make use
of TOAST functions for distributing data and then call built-in or custom operators to
process the timestream data.

The Time-Ordered Astrophysics Scalable Tools (TOAST) package is a software framework designed for simulation and reduction of data from telescope receivers which acquire timestreams of individual detector responses.  This type of instrumentation is often used when observing wavelengths from the microwave through the far IR.  Data from such telescopes present unique challenges compared to images acquired in optical astronomy.  Timestreams of detector responses can be correlated in time due to the optical response time of the detector and noise in the readout chain.  Timestreams can be correlated between detectors due to thermal fluctuations, readout cross-talk, atmospheric emission and pickup from the ground.  These correlations motivate us to process larger pieces of data simultaneously in order to extract the best possible estimate of the underlying signal coming from space, while reducing statistical errors and removing sources of systematic error.  To accommodate the collective processing of large data sets, TOAST has been designed to scale to high concurrency to handle the largest data volumes.  TOAST workflows have been run on systems ranging from a laptop up to 150,000 cores of the largest HPC system at the NERSC computing center.  The software has been developed over the course of 15 years and three major revisions.

The TOAST framework allows the user to build up a simulation and reduction workflow composed of well tested and general tools, as well as custom operations specific to a given experiment.  TOAST provides modules for simulating sky signal, correlated instrument noise, readout crosstalk, realistic atmospheric signal, beam asymmetries, ground pickup, calibration errors, pointing errors, and more.  For data reduction, the package provides modules for a variety of time domain and spatial filters, simple binned mapmaking, and a "generalized destriping" mapmaker which can regress templates from the data to model the noise, gain drifts, and other contaminating signals.  Along with these built-in modules, experiments can create their own simulation and reduction modules to build a workflow customized to a specific use case.

The TOAST framework has been used in previous experiments, including the [joint LFI/HFI Planck analysis](https://arxiv.org/abs/2007.04997).  The TOAST framework is one component of the data management tools being developed by the Simons Observatory.  TOAST is also part of the baseline plan for CMB-S4 simulation and reduction at HPC centers such as NERSC and ALCF.  Because TOAST is a modular framework, existing legacy tools from past experiments can be integrated into TOAST workflows or reimplemented as native TOAST modules.  Some examples of this are the ACT-style maximum likelihood mapmaker being developed for Simons Observatory Large Aperture Telescope data processing and the "observation matrix" approach used by BICEP/Keck that has been reimplemented in TOAST for small aperture telescopes.

## Specific Experiments

If you are a member of one of these projects:

- Planck
- LiteBIRD
- Simons Array
- Simons Observatory
- CMB-S4

Then there are additional software repositories you have access to that contain extra
TOAST classes and scripts for processing data from your experiment.
