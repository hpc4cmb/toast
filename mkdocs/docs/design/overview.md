# Overview


The TOAST framework contains:

- Tools for distributing data among many processes and performing collective operations using MPI.
- Tools for performing operations on the local pieces of the data
- Generic operators for common processing tasks (filtering, pointing expansion, map-making)
- Basic classes for performing I/O in a limited set of formats
- Well-defined interfaces for adding custom I/O classes and processing operators

The highest-level control of the workflow is done by the user, often by writing
a small Python script or notebook (some examples are included). Such scripts
make use of TOAST functionality for distributing data and then call built-in or
custom operators to simulate and / or process the timestream data.

## Data Distribution

- Domain decomposition



## Data Quality Flags

Detector data can be "flagged" or cut for a variety of reasons. Sometimes a flag
indicates that the data is corrupted in some way and should not be used. In
other cases, flags are used to classify data with particular properties. For
example, samples crossing a planet or while the telescope is in a turnaround.
TOAST supports flags that can be applied for an entire detector within an
observation, and also flags for individual detector samples.

## Pointing Model
