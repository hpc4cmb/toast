.. _utilsrng:

Random Number Generation
-----------------------------------

The following functions define wrappers around an internally-built version of the Random123 package for streamed random number generation.  This generator is fast and can return reproducible values from any location in the stream specified by the input key and counter values.

.. autofunction:: toast.rng.random

If generating samples from multiple different streams you can use this function:

.. autofunction:: toast.rng.random_multi
