# Data Model

- Domain decomposition



## Data Quality Flags

Detector data can be "flagged" or cut for a variety of reasons. Sometimes a flag
indicates that the data is corrupted in some way and should not be used. In
other cases, flags are used to classify data with particular properties. For
example, samples crossing a planet or while the telescope is in a turnaround.
TOAST supports flags that can be applied for an entire detector within an
observation, and also flags for individual detector samples.

## Pointing Model
