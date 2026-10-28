# Time Ordered Astrophysics Scalable Tools

[![Build Status](https://github.com/hpc4cmb/toast/actions/workflows/test.yml/badge.svg)](https://github.com/hpc4cmb/toast/actions/workflows/test.yml)

## Description

TOAST is a [software framework](https://en.wikipedia.org/wiki/Software_framework) for
simulating and processing timestream data collected by telescopes. Telescopes which
collect data as timestreams rather than images give us a unique set of analysis
challenges. Detector data usually contains noise which is correlated in time as well as
sources of correlated signal from the instrument and the environment. Large pieces of
data must often be analyzed simultaneously to extract an estimate of the sky signal.

## Documentation

See the full documentation in progress here:

https://hpc4cmb.github.io/toast

## Notice on AI Coding Assistants

Contributors to this project may make use of a variety of tools, including those that 
leverage AI.  These guidelines must be followed:

1. A human must submit every pull request and is responsible for communication 
surrounding the revisions of that pull request until it is merged.

2. The human contributor is responsible for ensuring that all code in a pull request 
does not violate existing licensing and copyright law.

3. The human contributor must understand and be able to explain how any submitted 
code functions.

4. Large-scale structural changes will receive extra scrutiny.  Design decisions will
be consciously chosen to enable the continued ability for humans to understand the 
codebase.
