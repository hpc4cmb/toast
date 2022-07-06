# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ._libtoast import (
    rng_dist_normal,
    rng_dist_uint64,
    rng_dist_uniform_01,
    rng_dist_uniform_11,
    rng_multi_dist_normal,
    rng_multi_dist_uint64,
    rng_multi_dist_uniform_01,
    rng_multi_dist_uniform_11,
)
from .dist import distribute_uniform
from .timing import function_timer
from .utils import AlignedF64, AlignedU64, Environment, Logger


@function_timer
def random(samples, key=(0, 0), counter=(0, 0), sampler="gaussian", threads=False):
    """Generate random samples from a distribution for one stream.

    This returns values from a single stream drawn from the specified
    distribution.  The starting state is specified by the two key values and
    the two counter values.  The second value of the "counter" is used to
    represent the sample index.  If the serial option is enabled, only a
    single thread will be used.  Otherwise the stream generation is divided
    equally between OpenMP threads.

    Args:
        samples (int): The number of samples to return.
        key (tuple): Two uint64 values which (along with the counter) define
            the starting state of the generator.
        counter (tuple): Two uint64 values which (along with the key) define
            the starting state of the generator.
        sampler (string): The distribution to sample from.  Allowed values are
            "gaussian", "uniform_01", "uniform_m11", and "uniform_uint64".
        threads (bool): If True, use OpenMP threads to generate the stream
            in parallel.  NOTE: this may actually run slower for short streams
            and many threads.

    Returns:
        (Aligned array): The random values of appropriate type for the sampler.

    """
    env = Environment.get()
    nthread = env.max_threads()
    log = Logger.get()
    ret = None
    if (not threads) or (samples < nthread):
        # Run serially
        if sampler == "gaussian":
            ret = AlignedF64(samples)
            rng_dist_normal(key[0], key[1], counter[0], counter[1], ret)
        elif sampler == "uniform_01":
            ret = AlignedF64(samples)
            rng_dist_uniform_01(key[0], key[1], counter[0], counter[1], ret)
        elif sampler == "uniform_m11":
            ret = AlignedF64(samples)
            rng_dist_uniform_11(key[0], key[1], counter[0], counter[1], ret)
        elif sampler == "uniform_uint64":
            ret = AlignedU64(samples)
            rng_dist_uint64(key[0], key[1], counter[0], counter[1], ret)
        else:
            msg = "Undefined sampler. Choose among: gaussian, uniform_01,\
                   uniform_m11, uniform_uint64"
            log.error(msg)
            raise ValueError(msg)
    else:
        # We are using threads, divide the samples up.
        dst = distribute_uniform(samples, nthread)
        k1 = AlignedU64(nthread)
        k2 = AlignedU64(nthread)
        c1 = AlignedU64(nthread)
        c2 = AlignedU64(nthread)
        k1[:] = np.array([key[0] for x in dst], dtype=np.uint64)
        k2[:] = np.array([key[1] for x in dst], dtype=np.uint64)
        c1[:] = np.array([counter[0] for x in dst], dtype=np.uint64)
        c2[:] = np.array([counter[1] + x[0] for x in dst], dtype=np.uint64)
        lengths = [x[1] for x in dst]

        if sampler == "gaussian":
            chunks = rng_multi_dist_normal(k1, k2, c1, c2, lengths)
            ret = AlignedF64(samples)
            for t in range(nthread):
                ret[dst[t][0] : dst[t][0] + dst[t][1]] = chunks[t]
        elif sampler == "uniform_01":
            chunks = rng_multi_dist_uniform_01(k1, k2, c1, c2, lengths)
            ret = AlignedF64(samples)
            for t in range(nthread):
                ret[dst[t][0] : dst[t][0] + dst[t][1]] = chunks[t]
        elif sampler == "uniform_m11":
            chunks = rng_multi_dist_uniform_11(k1, k2, c1, c2, lengths)
            ret = AlignedF64(samples)
            for t in range(nthread):
                ret[dst[t][0] : dst[t][0] + dst[t][1]] = chunks[t]
        elif sampler == "uniform_uint64":
            chunks = rng_multi_dist_uint64(k1, k2, c1, c2, lengths)
            ret = AlignedU64(samples)
            for t in range(nthread):
                ret[dst[t][0] : dst[t][0] + dst[t][1]] = chunks[t]
        else:
            msg = "Undefined sampler. Choose among: gaussian, uniform_01,\
                   uniform_m11, uniform_uint64"
            log.error(msg)
            raise ValueError(msg)
    return ret


@function_timer
def random_multi(samples, keys, counters, sampler="gaussian"):
    """Generate random samples from multiple streams.

    Given multiple streams, each specified by a pair of key values and a pair
    of counter values, generate some number of samples from each.  The number
    of samples is specified independently for each stream.  The generation of
    the streams is run with multiple threads.

    NOTE:  if you just want threaded generation of a single stream, use the
    "threads" option to the random() function.

    Args:
        samples (list): The number of samples to return for each stream
        keys (list): A tuple of integer values for each stream, which (along
            with the counter) define the starting state of the generator for
            that stream.
        counters (list): A tuple of integer values for each stream, which
            (along with the key) define the starting state of the generator
            for that stream.
        sampler (string): The distribution to sample from.  Allowed values are
            "gaussian", "uniform_01", "uniform_m11", and "uniform_uint64".

    Returns:
        (list): The random samples for each stream.

    """
    log = Logger.get()
    k1 = AlignedU64(len(keys))
    k2 = AlignedU64(len(keys))
    c1 = AlignedU64(len(counters))
    c2 = AlignedU64(len(counters))
    k1[:] = np.array([x[0] for x in keys], dtype=np.uint64)
    k2[:] = np.array([x[1] for x in keys], dtype=np.uint64)
    c1[:] = np.array([x[0] for x in counters], dtype=np.uint64)
    c2[:] = np.array([x[1] for x in counters], dtype=np.uint64)

    ret = None
    if sampler == "gaussian":
        ret = rng_multi_dist_normal(k1, k2, c1, c2, samples)
    elif sampler == "uniform_01":
        ret = rng_multi_dist_uniform_01(k1, k2, c1, c2, samples)
    elif sampler == "uniform_m11":
        ret = rng_multi_dist_uniform_11(k1, k2, c1, c2, samples)
    elif sampler == "uniform_uint64":
        ret = rng_multi_dist_uint64(k1, k2, c1, c2, samples)
    else:
        msg = "Undefined sampler. Choose among: gaussian, uniform_01,\
               uniform_m11, uniform_uint64"
        log.error(msg)
        raise ValueError(msg)
    return ret
