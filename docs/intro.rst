.. _intro:

Introduction
=================================

TOAST is a `software framework <https://en.wikipedia.org/wiki/Software_framework>`_ for simulating and processing timestream data collected by telescopes.
Telescopes which collect data as timestreams rather than images give us a unique set of analysis challenges.
Detector data usually contains noise which is correlated in time as well as sources of correlated signal from the instrument and the environment.
Large pieces of data must often be analyzed simultaneously to extract an estimate of the sky signal.
TOAST has evolved over several years.
The current codebase contains an internal C++ library to allow for optimization of some calculations, while the public interface is written in Python.

The TOAST framework contains:

    * Tools for distributing data among many processes
    * Tools for performing operations on the local pieces of the data
    * Generic operators for common processing tasks (filtering, pointing expansion, map-making)
    * Basic classes for performing I/O in a limited set of formats
    * Well-defined interfaces for adding custom I/O classes and processing operators

The highest-level control of the workflow is done by the user, often by writing a small Python "pipeline" script (some examples are included).  Such pipeline scripts make use of TOAST functions for distributing data and then call built-in or custom operators to process the timestream data.


Data Organization
-----------------------

The TOAST framework groups data into one or more "observations".  Each observation represents data from a group of detectors for some time span.  Detectors in the same observation must have the same number of samples for the length of the observation.  We currently also assume that the noise properties of the detectors are constant across this observation (i.e. the noise is stationary).  A TOAST "dataset" is simply a collection of one or more observations.


Workflow
---------------------


Example:  Satellite


Example:  Ground-Based



Support for Specific Experiments
-------------------------------------

If you are a member of one of these projects:

    * Planck
    * LiteBIRD
    * Simons Array
    * Simons Observatory
    * CMB-S4

Then there are additional software repositories you have access to that contain extra TOAST classes and scripts for processing data from your experiment.
