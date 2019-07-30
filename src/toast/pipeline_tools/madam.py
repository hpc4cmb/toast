# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import copy
import os
import re

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from ..map import OpMadam


def add_madam_args(parser, ground_data=True):
    """ Add libmadam arguments

    Args:
        ground_data (bool) :  If true, assume that Madam will be used to
            process ground experiment data and every process will have
            approximately equal sky coverage.
    """

    parser.add_argument(
        "--madam-prefix", required=False, default="toast", help="Output map prefix"
    )
    parser.add_argument(
        "--madam-iter-max",
        required=False,
        default=1000,
        type=np.int,
        help="Maximum number of CG iterations in Madam",
    )
    parser.add_argument(
        "--madam-baseline-length",
        required=False,
        default=10000.0,
        type=np.float,
        help="Destriping baseline length (seconds)",
    )
    parser.add_argument(
        "--madam-baseline-order",
        required=False,
        default=0,
        type=np.int,
        help="Destriping baseline polynomial order",
    )
    parser.add_argument(
        "--madam-noisefilter",
        required=False,
        default=False,
        action="store_true",
        help="Destripe with the noise filter enabled",
    )
    parser.add_argument(
        "--madam-parfile", required=False, default=None, help="Madam parameter file"
    )
    parser.add_argument(
        "--madam-allreduce",
        required=False,
        action="store_true",
        help="Use the allreduce commucation pattern in Madam",
        dest="madam_allreduce",
    )
    parser.add_argument(
        "--no-madam-allreduce",
        required=False,
        action="store_false",
        help="Do not use the allreduce commucation pattern in Madam",
        dest="madam_allreduce",
    )
    parser.set_defaults(madam_destripe=ground_data)

    parser.add_argument(
        "--destripe",
        required=False,
        action="store_true",
        help="Write Madam destriped maps",
        dest="madam_destripe",
    )
    parser.add_argument(
        "--no-destripe",
        required=False,
        action="store_false",
        help="Do not write Madam destriped maps",
        dest="madam_destripe",
    )
    parser.set_defaults(madam_destripe=True)

    parser.add_argument(
        "--binmap",
        required=False,
        action="store_true",
        help="Write Madam binned maps",
        dest="madam_binmap",
    )
    parser.add_argument(
        "--no-binmap",
        required=False,
        action="store_false",
        help="Do not write Madam binned maps",
        dest="madam_binmap",
    )
    parser.set_defaults(madam_binmap=True)

    parser.add_argument(
        "--hits",
        required=False,
        action="store_true",
        help="Write Madam hit maps",
        dest="madam_hits",
    )
    parser.add_argument(
        "--no-hits",
        required=False,
        action="store_false",
        help="Do not write Madam hit maps",
        dest="madam_hits",
    )
    parser.set_defaults(madam_hits=True)

    parser.add_argument(
        "--wcov",
        required=False,
        action="store_true",
        help="Write Madam white noise covariance",
        dest="madam_wcov",
    )
    parser.add_argument(
        "--no-wcov",
        required=False,
        action="store_false",
        help="Do not write Madam white noise covariance",
        dest="madam_wcov",
    )
    parser.set_defaults(madam_wcov=True)

    parser.add_argument(
        "--wcov-inv",
        required=False,
        action="store_true",
        help="Write Madam inverse white noise covariance",
        dest="madam_wcov_inv",
    )
    parser.add_argument(
        "--no-wcov-inv",
        required=False,
        action="store_false",
        help="Do not write Madam inverse white noise covariance",
        dest="madam_wcov_inv",
    )
    parser.set_defaults(madam_wcov_inv=True)

    parser.add_argument(
        "--conserve-memory",
        dest="conserve_memory",
        required=False,
        action="store_true",
        help="Conserve memory when staging libMadam buffers",
    )
    parser.add_argument(
        "--no-conserve-memory",
        dest="conserve_memory",
        required=False,
        action="store_false",
        help="Do not conserve memory when staging libMadam buffers",
    )
    parser.set_defaults(conserve_memory=True)

    # `nside` may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
        )
    except argparse.ArgumentError:
        pass
    # Common flag mask may already be added
    try:
        parser.add_argument(
            "--common-flag-mask",
            required=False,
            default=1,
            type=np.uint8,
            help="Common flag mask",
        )
    except argparse.ArgumentError:
        pass
    # `sample-rate` may be already added
    try:
        parser.add_argument(
            "--sample-rate",
            required=False,
            default=100.0,
            type=np.float,
            help="Detector sample rate (Hz)",
        )
    except argparse.ArgumentError:
        pass
    return


@function_timer
def setup_madam(args):
    """ Create a Madam parameter dictionary.

    Initialize the Madam parameters from the command line arguments.
    """
    pars = {}

    cross = args.nside // 2
    submap = 16
    if submap > args.nside:
        submap = args.nside

    pars["temperature_only"] = False
    pars["force_pol"] = True
    pars["kfirst"] = args.madam_destripe
    pars["write_map"] = args.madam_destripe
    pars["write_binmap"] = args.madam_binmap
    pars["write_matrix"] = args.madam_wcov_inv
    pars["write_wcov"] = args.madam_wcov
    pars["write_hits"] = args.madam_hits
    pars["nside_cross"] = cross
    pars["nside_submap"] = submap
    pars["allreduce"] = args.madam_allreduce
    pars["reassign_submaps"] = True
    pars["pixlim_cross"] = 1e-3
    pars["pixmode_cross"] = 2
    pars["pixlim_map"] = 1e-2
    pars["pixmode_map"] = 2
    # Instead of fixed detector weights, we'll want to use scaled noise
    # PSD:s that include the atmospheric noise
    pars["radiometers"] = True
    pars["noise_weights_from_psd"] = True

    if args.madam_parfile is not None:
        # Parse all available parameters from the supplied
        # Madam parameter file
        pat = re.compile(r"\s*(\S+)\s*=\s*(\S+(\s+\S+)*)\s*")
        comment = re.compile(r"^#.*")
        with open(args.madam_parfile, "r") as f:
            for line in f:
                if comment.match(line) is None:
                    result = pat.match(line)
                    if result is not None:
                        key, value = result.group(1), result.group(2)
                        pars[key] = value

    pars["base_first"] = args.madam_baseline_length
    pars["basis_order"] = args.madam_baseline_order
    pars["nside_map"] = args.nside
    if args.madam_noisefilter:
        if args.madam_baseline_order != 0:
            raise RuntimeError(
                "Madam cannot build a noise filter when baseline"
                "order is higher than zero."
            )
        pars["kfilter"] = True
    else:
        pars["kfilter"] = False
    pars["fsample"] = args.sample_rate
    pars["iter_max"] = args.madam_iter_max
    pars["file_root"] = args.madam_prefix
    return pars


@function_timer
def apply_madam(
    args,
    comm,
    data,
    madampars,
    mc,
    outpath,
    detweights,
    cache_name,
    freq=None,
    time_comms=None,
    telescope_data=None,
    first_call=True,
    extra_prefix=None,
    verbose=True,
    bin_only=False,
):
    """ Use libmadam to bin and optionally destripe data.

    Bin and optionally destripe all conceivable subsets of the data.

    Args:
        freq (str) :  Frequency identifier to append to the file prefix
        time_comms (iterable) :  Series of disjoint communicators that
            map, e.g., seasons and days.  Each entry is a tuple of
            the form (`name`, `communicator`)
        telescope_data (iterable) : series of disjoint TOAST data
            objects.  Each entry is tuple of the form (`name`, `data`).
        bin_only (bool) :  Disable destriping and only bin the signal,
            Useful when running Madam as a part of a filter+bin pipeline.
    """
    log = Logger.get()
    timer = Timer()
    timer.start()
    if comm.world_rank == 0 and verbose:
        log.info("Making maps")

    pars = copy.deepcopy(madampars)
    pars["path_output"] = outpath
    file_root = pars["file_root"]
    if extra_prefix is not None:
        if len(file_root) > 0 and not file_root.endswith("_"):
            file_root += "_"
        file_root += "{}".format(extra_prefix)
    if freq is not None:
        if len(file_root) > 0 and not file_root.endswith("_"):
            file_root += "_"
        file_root += "{:03}".format(int(freq))

    if first_call:
        # Only the first MC iteration should produce the hits and
        # white noise matrices
        pars["write_matrix"] = False
        pars["write_wcov"] = False
        pars["write_hits"] = False
    else:
        pars["write_matrix"] = False
        pars["write_wcov"] = False
        pars["write_hits"] = False

    if bin_only:
        pars["kfirst"] = False
        pars["write_map"] = False
        pars["write_binmap"] = True

    # Sanity check, is any of the Madam outputs required?

    outputs = [
        pars["write_map"],
        pars["write_binmap"],
        pars["write_hits"],
        pars["write_wcov"],
        pars["write_matrix"],
    ]
    if not np.any(outputs):
        if comm.world_rank == 0:
            log.info("No Madam outputs requested.  Skipping.")
        return

    if args.madam_noisefilter or not pars["kfirst"]:
        # With the noise filter enabled, we want to enforce continuity
        # across the Observation.  Otherwise we fit each interval
        # separately.
        madam_intervals = None
    else:
        madam_intervals = "intervals"

    madam = OpMadam(
        params=pars,
        detweights=detweights,
        name=cache_name,
        common_flag_mask=args.common_flag_mask,
        purge_tod=False,
        intervals=madam_intervals,
        conserve_memory=args.conserve_memory,
    )

    if "info" in madam.params:
        info = madam.params["info"]
    else:
        info = 3

    if time_comms is None:
        time_comms = [("all", comm.comm_world)]

    if telescope_data is None:
        telescope_data = ["all", data]

    ttimer = Timer()
    for time_name, time_comm in time_comms:
        for tele_name, tele_data in telescope_data:
            if len(time_name.split("-")) == 3:
                # Special rules for daily maps
                if args.skip_daymaps:
                    continue
                if len(telescope_data) > 1 and tele_name == "all":
                    # Skip daily maps over multiple telescopes
                    continue
                if first_call:
                    # Do not destripe daily maps
                    kfirst_save = pars["kfirst"]
                    write_map_save = pars["write_map"]
                    write_binmap_save = pars["write_binmap"]
                    pars["kfirst"] = False
                    pars["write_map"] = False
                    pars["write_binmap"] = True

            ttimer.start()
            madam.params["file_root"] = "{}_telescope_{}_time_{}".format(
                file_root, tele_name, time_name
            )
            if time_comm == comm.comm_world:
                madam.params["info"] = info
            else:
                # Cannot have verbose output from concurrent mapmaking
                madam.params["info"] = 0
            if time_comm.rank == 0 and verbose:
                log.info("Mapping {}".format(madam.params["file_root"]))
            madam.exec(tele_data, time_comm)

            time_comm.barrier()
            if comm.world_rank == 0 and verbose:
                ttimer.report_clear("Mapping {}".format(madam.params["file_root"]))

            if len(time_name.split("-")) == 3 and first_call:
                # Restore destriping parameters
                pars["kfirst"] = kfirst_save
                pars["write_map"] = write_map_save
                pars["write_binmap"] = write_binmap_save

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("Madam total")

    return
