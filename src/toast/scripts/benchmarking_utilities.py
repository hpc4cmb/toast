# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
Functions common to all benchmarking scripts.

total_sample = num_obs * obs_minutes * sample_rate * n_detector
"""

import copy
import json
import math
import os
from datetime import datetime

import healpy
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import QTable
from pkg_resources import resource_filename

import toast
import toast.ops
from toast.instrument import Focalplane
from toast.instrument_sim import fake_hexagon_focalplane
from toast.job import get_node_mem, job_size
from toast.timing import function_timer

from ..utils import Environment, Logger, Timer


def python_startup_time(rank):
    """Compute the python interpreter start time."""
    if rank == 0:
        log = toast.utils.Logger.get()
        if "TOAST_PYTHON_START" in os.environ:
            tinit = datetime.strptime(os.getenv("TOAST_PYTHON_START"), "%Y%m%d-%H%M%S")
            tnow = datetime.now()
            dt = tnow - tinit
            minutes = dt.seconds / 60.0
            msg = f"Python interpreter startup took {minutes:0.2f} minutes"
            log.info(msg)
        else:
            msg = f"To print python startup time, in the calling environment do:\n"
            msg += (
                f"            TOAST_PYTHON_START=$(date '+%Y%m%d-%H%M%S') <script> ..."
            )
            log.info(msg)


def get_mpi_settings(args, log, env):
    """
    Getting the MPI settings
    taking the dry_run parameter into account
    """
    # gets actual MPI information
    world_comm, procs, rank = toast.get_world()
    log.info_rank(f"TOAST version = {env.version()}", comm=world_comm)
    log.info_rank(
        f"Using a maximum of {env.max_threads()} threads per process", comm=world_comm
    )
    if world_comm is None:
        log.info_rank(
            f"Running serially with one process at {str(datetime.now())}",
            comm=world_comm,
        )
    else:
        log.info_rank(
            f"Running with {procs} processes at {str(datetime.now())}", comm=world_comm
        )

    # is this a dry run that does not use MPI
    if args.dry_run is not None:
        procs, procs_per_node = args.dry_run.split(",")
        procs = int(procs)
        procs_per_node = int(procs_per_node)
        log.info_rank(
            f"DRY RUN simulating {procs} total processes with {procs_per_node} per node",
            comm=world_comm,
        )
        # We are simulating the distribution
        min_avail, max_avail = get_node_mem(world_comm, 0)
        avail_node_bytes = max_avail
    else:
        # Get information about the actual job size
        (n_node, procs_per_node, min_avail, max_avail) = job_size(world_comm)
        avail_node_bytes = max_avail

    # sets per node memory
    log.info_rank(
        f"Minimum detected per-node memory available is {avail_node_bytes / (1024 ** 3) :0.2f} GB",
        comm=world_comm,
    )
    if args.node_mem_gb is not None:
        avail_node_bytes = int((1024**3) * args.node_mem_gb)
        log.info_rank(
            f"Setting per-node available memory to {avail_node_bytes / (1024 ** 3) :0.2f} GB as requested",
            comm=world_comm,
        )

    # computes the total number of nodes
    n_nodes = procs // procs_per_node
    log.info_rank(f"Job has {n_nodes} total nodes", comm=world_comm)

    return world_comm, procs, rank, n_nodes, avail_node_bytes


def memory_use(n_detector, group_nodes, total_samples, full_pointing):
    """Compute the memory use for a given configuration.

    Args:
        n_detector (int):  The number of detectors.
        group_nodes (int):  The number of nodes in one group.
        total_samples (int):  The total number of detector samples in the job.
        full_pointing (bool):  If True, we are storing full detector pointing in
            memory.

    Returns:
        (int):  The bytes of allocated memory.

    """
    # Number of detector samples
    det_samps = total_samples // n_detector

    # detector timestream, pixel index and 3 IQU weights
    detector_timestream_cost = (1 + 4) if full_pointing else 1

    det_bytes_per_sample = 2 * (  # At most 2 detector data copies.
        8 * detector_timestream_cost  # 64 bit float / ints used
        + 1  # one byte per sample for flags
    )

    common_bytes_per_sample = (
        8 * (4) * 2  # 64 bit floats x 2 boresight quaternions per sample
        + 8 * 3 * 2  # 64 bit floats x (position + velocity) vectors per sample
        + 1  # one byte per sample for common flag
    )

    bytes_per_samp = (
        n_detector * det_bytes_per_sample + group_nodes * common_bytes_per_sample
    )

    # Upper bound on non-acounted-for per sample overhead.
    # FIXME: we should track this down / refine eventually.
    bytes_per_samp *= 2

    return bytes_per_samp * det_samps


def select_distribution(
    n_nodes,
    n_procs,
    scans,
    max_n_detector,
    sample_rate,
    full_pointing,
    world_comm,
    per_process_overhead_bytes,
    max_samples=None,
    max_memory_bytes=None,
    target_proc_dets=20,
    force_group_nodes=None,
):
    """Choose a group size that load balances across both detectors and observations.

    The algorithm is as follows:  first use a single observation and one process group
    to increase the number of detectors to a maximum value while keeping the number of
    samples and memory use below the limits.

    If more data will fit in the sample / memory constraints, then use the max number
    of detectors and increase the number of observations one at a time.  For each
    increment in the number of observations, recompute the "best" group size within the
    sample / memory constraints.

    The "best" group size is one where each process has close to the target_proc_dets
    detectors in an observation, and each group has at least one observation.

    Args:
        n_nodes (int):  The number of nodes in the job.
        n_procs (int):  The number of MPI processes in the job.
        scans (list):  The list of observing scans.
        max_n_detector (int):  The maximum number of detectors.
        sample_rate (float):  The detector sample rate.
        full_pointing (bool):  If True, we are storing full detector pointing in
            memory.
        world_comm (mpi4py.Comm):  MPI communicator or None.
        per_process_overhead_bytes (int):  The memory overhead per process.
        max_samples (int):  The maximum number of samples or None.
        max_memory_bytes (int):  The maximum memory to use in bytes, or None.
        target_proc_dets (int):  The approximate number of detectors per process
            to attempt.
        force_group_nodes (int):  If not None, force the selection of the group
            size to be this number of nodes.

    Returns:
        (tuple):  The (group_nodes, memory_bytes) of the case with the smallest
            memory footprint.

    """
    log = toast.utils.Logger.get()

    if max_samples is None and max_memory_bytes is None:
        raise RuntimeError(
            "You must specify at least one of max_samples and max_memory_bytes"
        )

    if max_samples is None:
        max_samples = np.inf
    if max_memory_bytes is None:
        max_memory_bytes = np.inf

    # Number of processes per node
    node_procs = n_procs // n_nodes

    # Per-process memory overhead
    overhead_bytes = n_procs * per_process_overhead_bytes

    # Determine the number of detectors from the first observation

    memory_bytes = 0
    n_detector = 0
    total_samples = 0
    new_scans = list()
    group_nodes = n_nodes

    scan_samples = int(sample_rate * (scans[0].stop - scans[0].start).total_seconds())
    msg = f"First observation (0 of {len(scans)}), {scan_samples} samples per detector"
    log.verbose_rank(msg, comm=world_comm)
    while True:
        # Increment by whole pixels
        test_n_detector = n_detector + 2
        if test_n_detector > max_n_detector:
            msg = f"First observation, {test_n_detector} dets > {max_n_detector}, break"
            log.verbose_rank(msg, comm=world_comm)
            break
        test_samples = test_n_detector * scan_samples
        if test_samples > max_samples:
            msg = f"First observation, {test_samples} samples > {max_samples}, break"
            log.verbose_rank(msg, comm=world_comm)
            break
        # Compute memory use assuming one group
        test_bytes = overhead_bytes + memory_use(
            test_n_detector, group_nodes, test_samples, full_pointing
        )
        if test_bytes > max_memory_bytes:
            msg = f"First observation, {test_bytes} bytes > {max_memory_bytes}, break"
            log.verbose_rank(msg, comm=world_comm)
            break
        n_detector = test_n_detector
        total_samples = test_samples
        memory_bytes = test_bytes

    new_scans.append(copy.deepcopy(scans[0]))

    if n_detector < max_n_detector:
        # This means that we have a very small job and only a subset of detectors in one
        # observation will fit.
        return (n_detector, total_samples, group_nodes, memory_bytes, new_scans)
    log.verbose_rank(
        f"Using maximum number of detectors = {max_n_detector}", comm=world_comm
    )

    # We have the maximum number of detectors.  Now increase the number of
    # observations, one at a time.  As each additional observation is considered, find
    # the best group size and verify that the memory is within limits.
    for isc, sc in enumerate(scans):
        if isc == 0:
            # Already handled this above
            continue
        log.verbose_rank(
            f"Try appending observation {isc} of {len(scans)}:", comm=world_comm
        )
        scan_samples += int(sample_rate * (sc.stop - sc.start).total_seconds())
        test_samples = n_detector * scan_samples
        if test_samples > max_samples:
            msg = f"  {test_samples} samples > {max_samples}, break"
            log.verbose_rank(msg, comm=world_comm)
            break

        if force_group_nodes is not None:
            # We will only test the requested group size
            test_nodes_best = None
            test_bytes_best = None

            test_nodes = force_group_nodes
            test_bytes = overhead_bytes + memory_use(
                n_detector, test_nodes, test_samples, full_pointing
            )
            if test_bytes > max_memory_bytes:
                # Too much memory use
                msg = f"  test group nodes = {test_nodes}: "
                msg += f"{test_bytes} bytes larger than maximum ({max_memory_bytes})"
                log.verbose_rank(msg, comm=world_comm)
            else:
                test_nodes_best = test_nodes
                test_bytes_best = test_bytes
        else:
            test_nodes_best = None
            test_bytes_best = None
            for test_nodes in range(n_nodes, 0, -1):
                if n_nodes % test_nodes != 0:
                    # Not a whole number of groups
                    continue
                n_group = n_nodes // test_nodes
                if n_group > isc + 1:
                    # More groups than we have observations
                    msg = f"  test group nodes = {test_nodes} ({n_group} groups): "
                    msg += f"too many groups for {isc + 1} observations"
                    log.verbose_rank(msg, comm=world_comm)
                    continue
                group_procs = test_nodes * node_procs
                if test_nodes < n_nodes and group_procs * target_proc_dets < n_detector:
                    # This group is too small
                    msg = (
                        f"  test group nodes = {test_nodes} ({group_procs} processes): "
                    )
                    msg += (
                        f"group too small for {n_detector} dets and {target_proc_dets} "
                    )
                    msg += f"dets per process"
                    log.verbose_rank(msg, comm=world_comm)
                    continue
                test_bytes = overhead_bytes + memory_use(
                    n_detector, test_nodes, test_samples, full_pointing
                )
                if test_bytes > max_memory_bytes:
                    # Too much memory use
                    msg = f"  test group nodes = {test_nodes}: "
                    msg += (
                        f"{test_bytes} bytes larger than maximum ({max_memory_bytes})"
                    )
                    log.verbose_rank(msg, comm=world_comm)
                    continue
                msg = f"  test group nodes = {test_nodes}: "
                msg += f"accept with {test_bytes} total bytes"
                log.verbose_rank(msg, comm=world_comm)
                test_nodes_best = test_nodes
                test_bytes_best = test_bytes
        if test_nodes_best is None:
            # We failed to find any group size that works.  Likely this is due
            # to exceeding memory limits
            msg = f"  No valid group size found"
            log.verbose_rank(msg, comm=world_comm)
            break
        # At this point we were able to find a group size that works for this number of
        # scans
        group_nodes = test_nodes_best
        total_samples = test_samples
        memory_bytes = test_bytes_best
        new_scans.append(copy.deepcopy(sc))

    return (n_detector, total_samples, group_nodes, memory_bytes, new_scans)


def select_case(
    args,
    jobargs,
    n_procs,
    n_nodes,
    avail_node_bytes,
    full_pointing,
    world_comm,
    per_process_overhead_bytes=1024**3,
    target_proc_dets=200,
):
    """
    Selects the most appropriate case size given the memory available and number of
    nodes.  Sets total_samples, n_detector and group_nodes in args.

    One can set `per_process_overhead_bytes` (which defaults to 1GB) to define a number
    of bytes that will be consummed by each process, independently of the number of
    samples, when using case=`auto`.

    """
    log = toast.utils.Logger.get()

    # See if the user is overriding the group size
    force_group_nodes = None
    if jobargs.group_size is not None:
        n_group = n_procs // jobargs.group_size
        force_group_nodes = n_nodes // n_group

    # Compute the aggregate memory that is currently available
    available_memory_bytes = n_nodes * avail_node_bytes

    # Compare to the per-process overhead
    overhead_bytes = n_procs * per_process_overhead_bytes
    if overhead_bytes > available_memory_bytes:
        msg = f"Per-process memory overhead is {n_procs} x "
        msg += f"{per_process_overhead_bytes / (1024 ** 3) :0.2f} GB "
        msg += f"= {overhead_bytes / (1024 ** 3) :0.2f} GB, which "
        msg += f"is larger than the available memory "
        msg += f"({available_memory_bytes / (1024 ** 3) :0.2f} GB)."
        msg += f" Use fewer processes per node."
        log.error_rank(msg, comm=world_comm)
        # No need to raise the same error on every process
        if world_comm is None or world_comm.rank == 0:
            raise RuntimeError(msg)

    if args.case != "auto":
        # availaibles sizes
        cases_samples = {
            "heroic": 5000000000000,  # O(1000) TB RAM
            "xlarge": 500000000000,  # O(100) TB RAM
            "large": 50000000000,  # O(10) TB RAM
            "medium": 5000000000,  # O(1) TB RAM
            "small": 500000000,  # O(100) GB RAM
            "xsmall": 50000000,  # O(10) GB RAM
            "tiny": 5000000,  # O(1) GB RAM
        }
        # force use the case size suggested by the user
        max_samples = cases_samples[args.case]

        (
            args.n_detector,
            args.total_samples,
            args.group_nodes,
            memory_used_bytes,
            new_scans,
        ) = select_distribution(
            n_nodes,
            n_procs,
            args.schedule.scans,
            args.max_detector,
            args.sample_rate,
            full_pointing,
            world_comm,
            per_process_overhead_bytes=per_process_overhead_bytes,
            max_samples=max_samples,
            max_memory_bytes=None,
            target_proc_dets=target_proc_dets,
            force_group_nodes=force_group_nodes,
        )

        # Update the schedule to use only our subset of scans
        args.schedule.scans = new_scans

        args.n_group = n_nodes // args.group_nodes
        args.group_procs = n_procs // args.n_group

        msg = f"Distribution using:\n"
        msg += f"  {args.n_detector} detectors and {len(new_scans)} observations\n"
        msg += f"  {args.total_samples} total samples\n"
        msg += f"  {args.n_group} groups of {args.group_nodes} nodes with {args.group_procs} processes each\n"
        msg += f"  {memory_used_bytes / (1024 ** 3) :0.2f} GB predicted memory use\n"
        msg += f"  ('{args.case}' workflow size)"
        log.info_rank(msg, comm=world_comm)

        if args.n_detector < args.group_procs:
            msg = f"Only {args.n_detector} detectors for {args.group_procs} processes- "
            msg += "some processes will be idle!"
            log.warning_rank(msg, comm=world_comm)

        if memory_used_bytes >= available_memory_bytes:
            msg = f"The selected case, '{args.case}' might not fit in memory "
            log.warning_rank(msg, comm=world_comm)
    else:
        msg = f"Using automatic workflow size selection (case='auto') with "
        msg += f"{(per_process_overhead_bytes) / (1024 ** 3) :0.2f} GB reserved "
        msg += "for per process overhead."
        log.info_rank(
            msg,
            comm=world_comm,
        )

        # finds the number of samples that gets us closest to the available memory
        (
            args.n_detector,
            args.total_samples,
            args.group_nodes,
            memory_used_bytes,
            new_scans,
        ) = select_distribution(
            n_nodes,
            n_procs,
            args.schedule.scans,
            args.max_detector,
            args.sample_rate,
            full_pointing,
            world_comm,
            per_process_overhead_bytes=per_process_overhead_bytes,
            max_samples=None,
            max_memory_bytes=available_memory_bytes,
            target_proc_dets=target_proc_dets,
            force_group_nodes=force_group_nodes,
        )

        # Update the schedule to use only our subset of scans
        args.schedule.scans = new_scans

        args.n_group = n_nodes // args.group_nodes
        args.group_procs = n_procs // args.n_group

        msg = f"Distribution using:\n"
        msg += f"  {args.n_detector} detectors and {len(new_scans)} observations\n"
        msg += f"  {args.total_samples} total samples\n"
        msg += f"  {args.n_group} groups of {args.group_nodes} nodes with {args.group_procs} processes each\n"
        msg += f"  {memory_used_bytes / (1024 ** 3) :0.2f} GB predicted memory use "
        msg += f"  ({available_memory_bytes / (1024 ** 3) :0.2f} GB available)\n"
        msg += f"  ('{args.case}' workflow size)"
        log.info_rank(msg, comm=world_comm)


def estimate_memory_overhead(
    n_procs,
    n_nodes,
    sky_fraction,
    nside_solve,
    world_comm,
    nside_final=None,
    sim_atmosphere=None,
    ces_max_time=None,
    fov=None,
):
    """Estimate bytes of memory used per-process for objects besides timestreams.

    Args:
        n_procs (int):  The number of MPI processes in the job.
        n_nodes (int):  The number of nodes in the job.
        sky_fraction (float):  The fraction of the sky covered by one process.
        nside_solve (int):  The healpix NSIDE value for the solver.
        world_comm (MPI_Comm):  World communicator
        nside_final (int):  The healpix NSIDE value for the final binning.
        sim_atmosphere (Operator):  Atmospheric simulation operator
        ces_max_time (float):  Maximum length of an observation.
        fov (float):  Field of view in degrees

    Returns:
        (int):  The bytes used.

    """
    log = toast.utils.Logger.get()
    # Start with 1GB for everything else
    base = 1024**3
    # base = 0

    # Compute the bytes per pixel.  We have:
    #   hits (int64):  8 bytes
    #   noise weighted map (3 x float64):  24 bytes
    #   condition number map (1 x float64):  8 bytes
    #   diagonal covariance (6 x float64):  48 bytes
    #   solved map (3 x float64):  24 bytes
    bytes_per_pixel = 8 + 24 + 8 + 48 + 24

    n_pixel_solve = sky_fraction * 12 * nside_solve**2
    n_pixel_final = 0 if nside_final is None else sky_fraction * 12 * nside_final**2

    overhead = base + (n_pixel_solve + n_pixel_final) * bytes_per_pixel

    if sim_atmosphere is not None and sim_atmosphere.enabled:
        # Compute a pessimistic estimate of the size of an atmospheric realization
        if ces_max_time is None:
            raise RuntimeError(
                "Cannot calculate atmospheric overhead without CES max time"
            )
        if fov is None:
            raise RuntimeError("Cannot calculate atmospheric overhead without FOV")
        zmax = sim_atmosphere.zmax.to_value(u.m)
        # Volume element size
        xstep = sim_atmosphere.xstep.to_value(u.m)
        ystep = sim_atmosphere.ystep.to_value(u.m)
        zstep = sim_atmosphere.zstep.to_value(u.m)
        # Boresight elevation
        elevation = np.radians(45)
        # Field of view
        radius = np.radians(fov) / 2
        windspeed = 10  # m/s
        # Maximum distance
        rmax = 100
        # Scale factor between simulations
        scale = 10
        # Jump straight to the third (largest) iteration
        rmax *= scale**2
        xstep *= scale
        ystep *= scale
        zstep *= scale
        # Height of the slab
        zmax = min(zmax, rmax * np.sin(elevation + radius))
        # Length of the slab
        xmax = rmax * np.cos(elevation - radius)
        # Width of the slab
        ymax = min(sim_atmosphere.wind_dist.to_value(u.m), ces_max_time * windspeed)
        n_elem_tot = xmax * ymax * zmax / (xstep * ystep * zstep)
        # Maximum memory is allocated when building the boolean
        # compression table.  Assume 1 byte per bool
        max_mem = n_elem_tot * 1
        if overhead - base < max_mem:
            msg = f"WARNING: Atmospheric overhead ({max_mem / 2 ** 30:.3f} GB) "
            msg += "exceeds local map overhead ({(overhead - base) / 2 ** 30:.3f} GB)."
            log.warning_rank(msg, comm=world_comm)
            overhead = base + max_mem

    return overhead


def get_standard_ground_args(args):
    # Sample rate in Hz
    args.sample_rate = 100

    args.max_detector = 2054
    if args.max_n_det is not None:
        args.max_detector = args.max_n_det

    # Width of the hexagon focalplane in degrees
    args.width = 10.0

    # Detector NET in K*sqrt(s)
    args.psd_net = 50.0e-6

    # Detector frequency of low-f rolloff
    args.psd_fmin = 1.0e-5

    # scan map
    args.nside = 4096
    args.input_map = f"fake_input_sky_nside{args.nside}.fits"

    # schedule, site and telescope parameters
    args.telescope_name = "LAT"
    args.site_name = "atacama"
    args.site_lon = " -67:47:10"
    args.site_lat = " -22:57:30"
    args.site_alt = 5200.0 * u.meter
    args.patch_coord = "C"
    args.el_min = 30.0
    args.el_max = 70.0
    args.sun_el_max = 90.0
    args.sun_avoidance_angle = 0.0
    args.moon_avoidance_angle = 0.0
    args.gap_s = 60.0
    args.gap_small_s = 0.0
    args.ces_max_time = 1200.0
    args.boresight_angle_step = 180.0
    args.boresight_angle_time = 1440.0
    args.schedule_patches = [
        "RISING_SCAN_35,HORIZONTAL,1.00,30.00,150.00,35.00,1500",
        "SETTING_SCAN_35,HORIZONTAL,1.00,210.00,330.00,35.00,1500",
    ]
    args.schedule_start = "2027-01-01 00:00:00"

    # This length should be sufficient for the "x-large" case.
    # args.schedule_stop = "2027-04-01 00:00:00"

    # This produces about 26000 scans, and is sufficient for even the "heroic" case:
    args.schedule_stop = "2027-12-31 00:00:00"

    # This produces 75 scans:
    # args.schedule_stop = "2027-01-01 01:00:00"


def make_focalplane(args, world_comm, log):
    """
    Creates a fake focalplane
    """
    # computes the number of pixels to be used
    ring = math.ceil(math.sqrt((args.n_detector - 2) / 6)) if args.n_detector > 2 else 0
    n_pixel = 1 + 3 * ring * (ring + 1)

    # creates the focalplane
    focalplane = None
    if (world_comm is None) or (world_comm.rank == 0):
        focalplane = fake_hexagon_focalplane(
            n_pix=n_pixel,
            width=args.width * u.degree,
            sample_rate=args.sample_rate * u.Hz,
            psd_net=args.psd_net * u.K * np.sqrt(1 * u.second),
            psd_fmin=args.psd_fmin * u.Hz,
        )
        if n_pixel != 2 * args.n_detector:
            # Truncate number of detectors to our desired value
            trunc_dets = QTable(focalplane.detector_data[0 : args.n_detector])
            focalplane = Focalplane(
                detector_data=trunc_dets,
                sample_rate=args.sample_rate * u.Hz,
                field_of_view=focalplane.field_of_view,
            )

    if world_comm is not None:
        focalplane = world_comm.bcast(focalplane, root=0)
    log.info_rank(
        f"Using {len(focalplane.detectors)//2} hexagon-packed pixels.", comm=world_comm
    )
    return focalplane


def create_input_maps(
    input_map_path, nside, rank, log, should_print_input_map_png=False
):
    """
    Creates a *completely* fake map for scan_map
    (just to have something on the sky besides zeros)
    puts it at input_map_path
    """
    if os.path.isfile(input_map_path) or (rank != 0):
        return
    log.info(f"Generating input map {input_map_path}")

    ell = np.arange(3 * nside - 1, dtype=np.float64)

    sig = 50.0
    numer = ell - 30.0
    tspec = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * numer**2 / sig**2)
    tspec *= 2000.0

    sig = 100.0
    numer = ell - 500.0
    espec = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * numer**2 / sig**2)
    espec *= 1.0

    cls = (
        tspec,
        espec,
        np.zeros(3 * nside - 1, dtype=np.float32),
        np.zeros(3 * nside - 1, dtype=np.float32),
    )
    np.random.seed(123456789)
    maps = healpy.synfast(
        cls,
        nside,
        pol=True,
        pixwin=False,
        sigma=None,
        new=True,
        fwhm=np.radians(3.0 / 60.0),
    )
    healpy.write_map(
        input_map_path,
        maps,
        nest=True,
        column_units="K",
        fits_IDL=False,
        dtype=np.float32,
    )

    # displays the map as a picture on file
    if should_print_input_map_png:
        healpy.mollview(maps[0])
        plt.savefig(f"{input_map_path}_fake-T.png")
        plt.close()

        healpy.mollview(maps[1])
        plt.savefig(f"{input_map_path}_fake-E.png")
        plt.close()


def scan_map(args, rank, job_ops, data, log):
    """
    Simulate sky signal from a map.
    We scan the sky with the "final" pointing model if that is different from the solver pointing model.
    """
    if job_ops.scan_map.enabled:
        # Use the final pointing model if it is enabled
        pixels = job_ops.pixels
        weights = job_ops.weights
        if job_ops.pixels_final.enabled:
            pixels = job_ops.pixels_final
        # creates a map and puts it in args.input_map
        create_input_maps(args.input_map, pixels.nside, rank, log, args.print_input_map)
        job_ops.scan_map.pixel_dist = job_ops.binner_final.pixel_dist
        job_ops.scan_map.pixel_pointing = pixels
        job_ops.scan_map.stokes_weights = weights
        job_ops.scan_map.save_pointing = job_ops.binner_final.full_pointing
        job_ops.scan_map.file = args.input_map
        job_ops.scan_map.apply(data)


def default_sim_atmosphere():
    """Return a SimAtmosphere operator with fixed defaults."""
    return toast.ops.SimAtmosphere(
        name="sim_atmosphere",
        lmin_center=0.001 * u.meter,
        lmin_sigma=0.0 * u.meter,
        lmax_center=1.0 * u.meter,
        lmax_sigma=0.0 * u.meter,
        gain=1.0e-4,
        zatm=40000 * u.meter,
        zmax=200 * u.meter,
        xstep=10 * u.meter,
        ystep=10 * u.meter,
        zstep=10 * u.meter,
        nelem_sim_max=10000,
        wind_dist=3000 * u.meter,
        z0_center=2000 * u.meter,
        z0_sigma=0 * u.meter,
    )


def run_mapmaker(job_ops, args, tmpls, data):
    """
    Build up our map-making operation from the pieces- both operators configured from user options and other operators.
    """

    job_ops.binner.noise_model = job_ops.default_model.noise_model
    job_ops.binner_final.noise_model = job_ops.default_model.noise_model

    job_ops.mapmaker.binning = job_ops.binner
    job_ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(
        templates=[tmpls.baselines],
        view=job_ops.pixels_final.view,
    )
    job_ops.mapmaker.map_binning = job_ops.binner_final
    job_ops.mapmaker.det_data = job_ops.sim_noise.det_data
    job_ops.mapmaker.output_dir = args.out_dir

    # Run the map making
    job_ops.mapmaker.apply(data)


def run_madam(job_ops, args, tmpls, data):
    """
    Apply the Madam mapmaker using TOAST mapmaker configuration
    """

    job_ops.mapmaker.binning = job_ops.binner
    job_ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(
        templates=[tmpls.baselines],
        view=job_ops.pixels_final.view,
    )
    job_ops.mapmaker.map_binning = job_ops.binner_final
    job_ops.mapmaker.output_dir = args.out_dir

    job_ops.madam.params = toast.ops.madam_params_from_mapmaker(job_ops.mapmaker)
    job_ops.madam.pixel_pointing = job_ops.pixels_final
    job_ops.madam.stokes_weights = job_ops.weights
    job_ops.madam.apply(data)


def compute_science_metric(args, runtime, n_nodes, rank, log):
    """Return the samples per node-second processed by a benchmark run.

    NOTE:  This number can only be used to compare workflows whose configuration
    options are identical aside from the data volume.

    """
    metric = args.total_samples / (n_nodes * runtime)
    if rank == 0:
        msg = f"Science Metric (samples per node-second):  "
        msg += f"({args.total_samples:0.3e}) / ({runtime:0.1f} * {n_nodes})"
        msg += f" = {metric:0.2f}"
        log.info("")
        log.info(msg)
        log.info("")
        with open(os.path.join(args.out_dir, "log"), "a") as f:
            f.write(msg)
            f.write("\n\n")


benchmark_stats_data = None


def get_benchmark_stats(jobtype, case):
    """Load bundled benchmark stats and return the specified case

    Args:
        jobtype (str):  ground, satellite, etc
        case (str):  small, medium, large, etc

    Returns:
        (dict):  The stats for this job.

    """
    global benchmark_stats_data

    if benchmark_stats_data is None:
        # Load the data
        benchmark_stats_data = dict()
        bench_dir = resource_filename("toast", os.path.join("aux", "benchmarks"))
        bench_file = os.path.join(bench_dir, "stats.json")
        if not os.path.isfile(bench_file):
            msg = f"benchmark stats file {bench_file} does not exist"
            raise RuntimeError(msg)
        with open(bench_file, "r") as f:
            stats = json.load(f)
            benchmark_stats_data.update(stats)

    if jobtype in benchmark_stats_data:
        if case in benchmark_stats_data[jobtype]:
            return benchmark_stats_data[jobtype][case]
    return None


def compare_output_stats(jobname, args, rank, log, out_hits, out_map):
    """Compare job outputs to bundled versions."""

    hit_stats = out_hits.stats()
    map_stats = out_map.stats()
    comp = get_benchmark_stats(jobname, args.case)
    if rank == 0:
        result = {
            "totalhits": int(hit_stats["sum"][0]),
            "rms_I": map_stats["rms"][0],
            "mean_Q": map_stats["mean"][1],
            "rms_Q": map_stats["rms"][1],
            "mean_U": map_stats["mean"][2],
            "rms_U": map_stats["rms"][2],
        }
        if comp is None:
            msg = f"Output statistics for case '{args.case}':\n"
            msg += f"  Total map hits = {result['totalhits']}\n"
            msg += f"  Intensity map RMS = {result['rms_I']}\n"
            msg += f"  Stokes Q map RMS = {result['rms_Q']}\n"
            msg += f"  Stokes U map RMS = {result['rms_U']}"
        else:
            msg = f"Output statistics for case '{args.case}':\n"
            msg += f"  Total map hits = {result['totalhits']} "
            msg += f"(expected {int(comp['totalhits'])})\n"
            msg += f"  Intensity map RMS = {result['rms_I']} "
            msg += f"(expected {comp['rms_I']})\n"
            msg += f"  Stokes Q map RMS = {result['rms_Q']} "
            msg += f"(expected {comp['rms_Q']})\n"
            msg += f"  Stokes U map RMS = {result['rms_U']} "
            msg += f"(expected {comp['rms_U']})"
        log.info("")
        log.info(msg)
        log.info("")
        with open(os.path.join(args.out_dir, "log"), "a") as f:
            f.write(msg)
            f.write("\n\n")
        # Dump out to json for easy combining later
        dump_result = {jobname: {args.case: result}}
        with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
            json.dump(dump_result, f)
