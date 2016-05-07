.. _intro:

Introduction
=================================

Telescopes which collect data as timestreams rather than images give us a unique set of analysis challenges.  Detector data usually contains noise which is correlated in time and sources of correlated signal from the instrument and the environment.  Large pieces of data must often be analyzed simultaneously to extract an estimate of the sky signal.  TOAST 2.0 evolved as a re-implementation (in Python) of an earlier codebase written in C++.  This was a pragmatic choice given the need to interface better with instrument scientists, and was made possible by improving support for Python on HPC systems.

TOAST is a Python package with tools for:

    * Distributing data among many processes
    * Performing operations on the local pieces of the data
    * Creating customized processing operations

This package comes with a set of basic operations that will expand as development continues.  All of the experiment-specific classes and pipeline scripts are kept in separate git repositories.  Currently repositories exist for:

    * Planck
    * LiteBIRD
    * CMB "Stage 4"

This list will grow over time.


Data Organization and Terminology
---------------------------------------


Example:  Satellite


Example:  Ground-Based


Example:  Balloon

