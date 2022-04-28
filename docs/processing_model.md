(procmodel:)=
# Processing Model

A TOAST workflow usually consists of loading or simulating different
types of data into one or more [Observation]{.title-ref} objects (See
section `observations`{.interpreted-text role="ref"}) inside an overall
[Data]{.title-ref} instance. The workflow classes that populate and
manipulate these Data objects are called \"Operators\".

## Operators

An operator class inherits from the [toast.ops.Operator]{.title-ref}
class, and has several key characteristics:

- Each Operator is configured using the [traitlets package](https://github.com/ipython/traitlets). This allows easy configuration of an operator at construction time or afterwards, and allows modular documentation of parameters, parameter checking, and construction from parameter files.

- Operators can be called repeatedly on subsets of data (both observations and detectors) with the [exec()]{.title-ref} method.

- Operators have a [finalize()]{.title-ref} method that performs any
     final calculations or other steps after all timestream data has
     been processed.

```{eval-rst}
.. autoclass:: toast.ops.Operator
    :members:
```

For details about specific operators see the relevant sections
(`simulation-operators`{.interpreted-text role="ref"},
`reduction-operators`{.interpreted-text role="ref"},
`utility-operators`{.interpreted-text role="ref"}).

## Pipeline Operator

Although one can run a single operator on the whole dataset before
running the next operator, a common processing paradigm is to run a
sequence of operations on one detector at a time or on sets of
detectors.

## Operator Configuration
