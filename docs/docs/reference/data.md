# Data Containers

TOAST uses several key classes for storing time ordered data, instrument properties, and pixel domain data.  TOAST supports fully distributing time and pixel domain data across multiple nodes of a cluster or supercomputer.

## Data Object

::: toast.mpi.Comm

::: toast.Data

## Observation

::: toast.observation.Observation

::: toast.observation.DistDetSamp

::: toast.observation.IntervalList
::: toast.observation.IntervalsManager

::: toast.observation.SharedDataManager
::: toast.observation.DetDataManager

::: toast.observation.ViewInterface

::: toast.observation.Session

Each `Observation` has its own instrument model (see Instrument Model section).


