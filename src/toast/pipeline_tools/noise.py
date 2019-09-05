# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from ..tod import AnalyticNoise, OpSimNoise


def add_noise_args(parser):
    """ Add the noise simulation arguments
    """
    parser.add_argument(
        "--noise",
        required=False,
        action="store_true",
        help="Add simulated noise",
        dest="simulate_noise",
    )
    parser.add_argument(
        "--simulate-noise",
        required=False,
        action="store_true",
        help="Add simulated noise",
        dest="simulate_noise",
    )
    parser.add_argument(
        "--no-noise",
        required=False,
        action="store_false",
        help="Do not add simulated noise",
        dest="simulate_noise",
    )
    parser.add_argument(
        "--no-simulate-noise",
        required=False,
        action="store_false",
        help="Do not add simulated noise",
        dest="simulate_noise",
    )
    parser.set_defaults(simulate_noise=False)
    return


@function_timer
def simulate_noise(
    args, comm, data, mc, cache_prefix=None, verbose=True, overwrite=False
):
    """ Simulate electronic noise
    """
    if not args.simulate_noise:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()
    if comm.world_rank == 0 and verbose:
        log.info("Simulating noise")
    if overwrite:
        # Clear existing signal from the cache
        for obs in data.obs:
            tod = obs["tod"]
            if cache_prefix is None:
                prefix = tod.SIGNAL_NAME
            else:
                prefix = cache_prefix
            tod.cache.clear(prefix + "_.*")
    nse = OpSimNoise(out=cache_prefix, realization=mc)
    nse.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Simulate noise")
    return


@function_timer
def get_analytic_noise(args, comm, focalplane, verbose=True):
    """Create a TOAST noise object.

    Create a noise object from the 1/f noise parameters contained in the
    focalplane database.

    """
    timer = Timer()
    timer.start()
    detectors = sorted(focalplane.keys())
    fmin = {}
    fknee = {}
    alpha = {}
    NET = {}
    rates = {}
    for d in detectors:
        rates[d] = args.sample_rate
        fmin[d] = focalplane[d]["fmin"]
        fknee[d] = focalplane[d]["fknee"]
        alpha[d] = focalplane[d]["alpha"]
        NET[d] = focalplane[d]["NET"]
    noise = AnalyticNoise(
        rate=rates, fmin=fmin, detectors=detectors, fknee=fknee, alpha=alpha, NET=NET
    )
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Creating noise model")
    return noise
