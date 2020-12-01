# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from .atm import (
    add_atmosphere_args,
    simulate_atmosphere,
    scale_atmosphere_by_frequency,
    update_atmospheric_noise_weights,
)
from .binning import add_binner_args, init_binner, apply_binner
from .debug import add_debug_args
from .dipole import add_dipole_args, simulate_dipole
from .dist import add_dist_args, get_comm, get_time_communicators
from .export import add_tidas_args, output_tidas, add_spt3g_args, output_spt3g
from .filters import (
    add_polyfilter_args,
    add_polyfilter2D_args,
    apply_polyfilter,
    apply_polyfilter2D,
    add_groundfilter_args,
    apply_groundfilter,
)
from .gain import add_gainscrambler_args, scramble_gains
from .mapmaker import add_mapmaker_args, apply_mapmaker
from .madam import add_madam_args, setup_madam, apply_madam
from .noise import add_noise_args, simulate_noise, get_analytic_noise
from .pointing import add_pointing_args, expand_pointing
from .sky_signal import (
    add_sky_map_args,
    add_pysm_args,
    scan_sky_signal,
    simulate_sky_signal,
    add_conviqt_args,
    apply_conviqt,
    apply_weighted_conviqt,
)
from .sss import add_sss_args, simulate_sss
from .todground import (
    add_todground_args,
    get_elevation_noise,
    get_breaks,
    load_schedule,
    load_weather,
    Site,
    CES,
    Schedule,
)
from .todsatellite import add_todsatellite_args


def add_mc_args(parser):
    """Add Monte Carlo arguments"""
    parser.add_argument(
        "--MC-start",
        required=False,
        default=0,
        type=np.int,
        help="First Monte Carlo noise realization",
    )
    parser.add_argument(
        "--MC-count",
        required=False,
        default=1,
        type=np.int,
        help="Number of Monte Carlo noise realizations",
    )
    return


@function_timer
def add_signal(args, comm, data, prefix_out, prefix_in, purge=False, verbose=True):
    """Add signal from cache prefix `prefix_in` to cache prefix
    `prefix_out`.  If `prefix_out` does not exit, it is created.

    """
    if prefix_in == prefix_out or prefix_in is None or prefix_out is None:
        return
    log = Logger.get()
    if comm.world_rank == 0 and verbose:
        log.info("Adding signal from {} to {}." "".format(prefix_in, prefix_out))
        if purge:
            log.info("Purging {} after adding".format(prefix_in))
    timer = Timer()
    timer.start()
    for obs in data.obs:
        tod = obs["tod"]
        for det in tod.local_dets:
            cachename_in = "{}_{}".format(prefix_in, det)
            cachename_out = "{}_{}".format(prefix_out, det)
            ref_in = tod.cache.reference(cachename_in)
            if tod.cache.exists(cachename_out):
                ref_out = tod.cache.reference(cachename_out)
                ref_out += ref_in
            else:
                ref_out = tod.cache.put(cachename_out, ref_in)
            del ref_in, ref_out
        # Purge only after all detectors are added, just in case
        # any one is an alias
        if purge:
            for det in tod.local_dets:
                cachename_in = "{}_{}".format(prefix_in, det)
                tod.cache.clear(cachename_in)
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Add signal")
    return


@function_timer
def copy_signal(args, comm, data, cache_prefix_in, cache_prefix_out, verbose=True):
    """Copy the signal in `cache_prefix_in` to `cache_prefix_out`."""
    if cache_prefix_in == cache_prefix_out:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()
    if comm.world_rank == 0 and verbose:
        log.info(
            "Copying signal from {} to {}" "".format(cache_prefix_in, cache_prefix_out)
        )
    cachecopy = OpCacheCopy(cache_prefix_in, cache_prefix_out, force=True)
    cachecopy.exec(data)
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Copy signal")
    return
