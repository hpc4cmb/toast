# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
Functions common to all benchmarking scripts.

total_sample = num_obs * obs_minutes * sample_rate * n_detector
"""

import math
import copy
import healpy
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import QTable
import toast
import toast.ops
from toast.job import job_size, get_node_mem
from toast.instrument import Focalplane
from toast.instrument_sim import fake_hexagon_focalplane


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
        avail_node_bytes = int((1024 ** 3) * args.node_mem_gb)
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

    # Upper bound on non-acounted-for per sample overhead
    bytes_per_samp *= 2

    return bytes_per_samp * (total_samples // n_detector)


def get_minimum_memory_use(
    n_detector, n_nodes, n_procs, total_samples, scans, full_pointing
):
    """Compute the group size that minimizes the aggregate memory use.

    Search the allowable values of the group size for the one which results in the
    smallest memory use.

    Args:
        n_detector (int):  The number of detectors.
        n_nodes (int):  The number of nodes in the job.
        n_procs (int):  The number of MPI processes in the job.
        total_samples (int):  The total number of detector samples in the job.
        scans (list):  The list of observing scans.
        full_pointing (bool):  If True, we are storing full detector pointing in
            memory.

    Returns:
        (tuple):  The (group_nodes, memory_bytes) of the case with the smallest
            memory footprint.

    """
    log = toast.utils.Logger.get()

    # The number of observations in the schedule
    num_obs = len(scans)

    # The number of processes per node
    node_procs = n_procs // n_nodes

    group_nodes_best = 0
    memory_used_bytes_best = np.inf

    # what is the minimum memory we can use for the total number of samples?
    for group_nodes in range(1, n_nodes + 1):
        if n_nodes % group_nodes == 0:
            # This is a valid group size.
            n_group = n_nodes // group_nodes
            if n_group > num_obs:
                # Too many small groups- we do not have enough observations to give at
                # least one to each group.
                msg = f"Rejecting possible group nodes = {group_nodes}, "
                msg += f"since {n_group} groups is larger than the number of "
                msg += f"observations ({num_obs})"
                log.verbose_rank(msg)
                continue
            group_procs = node_procs * group_nodes
            if group_procs > n_detector:
                # This group is too large for the number of detectors
                msg = f"Rejecting possible group nodes = {group_nodes}, "
                msg += f"since {group_procs} processes per group is larger "
                msg += f"than the number of detectors ({n_detector})"
                log.verbose_rank(msg)
                continue

            memory_used_bytes = memory_use(
                n_detector, group_nodes, total_samples, full_pointing
            )

            if memory_used_bytes < memory_used_bytes_best:
                group_nodes_best = group_nodes
                memory_used_bytes_best = memory_used_bytes

    return (group_nodes_best, memory_used_bytes_best)


def maximize_nb_samples(
    n_nodes,
    n_procs,
    scans,
    max_n_detector,
    sample_rate,
    full_pointing,
    available_memory_bytes,
    per_process_overhead_bytes=1024 ** 3,
):
    """Finds the largest number of samples that can fit in the available memory.

    Return the resulting number of detectors, group size, total number of
    samples, total memory use, and the list of observing scans.

    One can set `per_process_overhead_bytes` (which defaults to 1GB) to define a number
    of bytes that will be consumed by each process, independently of the number of
    samples.

    Args:
        n_nodes (int):  The number of nodes in the job.
        n_procs (int):  The number of MPI processes in the job.
        scans (list):  The list of observing scans.
        max_n_detector (int):  The maximum number of detectors.
        sample_rate (float):  The detector sample rate.
        full_pointing (bool):  If True, we are storing full detector pointing in
            memory.
        available_memory_bytes (int):  The total aggregate memory in the job.
        per_process_overhead_bytes (int):  The memory overhead per process.

    Returns:
        (tuple):  The (n_detector, new_scans, total_samples, group_nodes, memory_bytes)
            of the best configuration.

    """
    log = toast.utils.Logger.get()

    # The output set of observation scans.
    new_scans = list()

    # The number of detectors.  Start with at least enough
    # detectors for the case of group_nodes == 1
    n_detector = n_procs // n_nodes

    # The total samples
    total = 0

    # The process group size
    group_nodes = 0

    # The estimated memory size of the configuration
    overhead_bytes = n_procs * per_process_overhead_bytes
    memory_bytes = 0

    scan_samples = 0
    for isc, sc in enumerate(scans):
        scan_samples += int(sample_rate * (sc.stop - sc.start).total_seconds())
        det_samps = n_detector * scan_samples
        if total == 0:
            # First scan, compute number of detectors
            while (
                n_detector < max_n_detector
                and (memory_bytes + overhead_bytes) < available_memory_bytes
            ):
                # Increment by whole pixels
                n_detector += 2
                det_samps = n_detector * scan_samples
                group_nodes, memory_bytes = get_minimum_memory_use(
                    n_detector,
                    n_nodes,
                    n_procs,
                    det_samps,
                    scans[: isc + 1],
                    full_pointing,
                )
                if group_nodes == 0:
                    # This distribution failed, ignore the returned memory use for the
                    # next loop iteration
                    memory_bytes = 0
            if group_nodes == 0:
                msg = f"At maximum detector count ({n_detector}), no compatible "
                msg += f"group size could be found for {n_procs} processes "
                msg += f"across {n_nodes} nodes"
                raise RuntimeError(msg)
            msg = f"Examining first observation, now using {n_detector} detectors"
            log.debug_rank(msg)
            total = det_samps
            new_scans.append(copy.deepcopy(sc))
        else:
            gs, bytes = get_minimum_memory_use(
                n_detector, n_nodes, n_procs, det_samps, scans[: isc + 1], full_pointing
            )
            if gs == 0:
                msg = f"For {n_detector} detectors and {det_samps} samples, "
                msg += f"no compatible group size could be found for "
                msg += f"{n_procs} processes across {n_nodes} nodes"
                raise RuntimeError(msg)
            if (bytes + overhead_bytes) > available_memory_bytes:
                break
            else:
                group_nodes = gs
                memory_bytes = bytes
                total = det_samps
                new_scans.append(copy.deepcopy(sc))

    memory_bytes += overhead_bytes

    return (n_detector, new_scans, total, group_nodes, memory_bytes)


def get_from_samples(
    n_nodes,
    n_procs,
    scans,
    max_n_detector,
    sample_rate,
    full_pointing,
    max_samples,
    per_process_overhead_bytes=1024 ** 3,
):
    """Finds the best configuration for a fixed number of samples.

    Similar to `maximize_nb_samples()`, but finds the instrument and observing
    configuration which fits within the requested number of samples.

    Return the resulting number of detectors, group size, total number of
    samples, total memory use, and the list of observing scans.

    One can set `per_process_overhead_bytes` (which defaults to 1GB) to define a number
    of bytes that will be consumed by each process, independently of the number of
    samples.

    Args:
        n_nodes (int):  The number of nodes in the job.
        n_procs (int):  The number of MPI processes in the job.
        scans (list):  The list of observing scans.
        max_n_detector (int):  The maximum number of detectors.
        sample_rate (float):  The detector sample rate.
        full_pointing (bool):  If True, we are storing full detector pointing in
            memory.
        max_samples (int):  The maximum number of samples.
        per_process_overhead_bytes (int):  The memory overhead per process.

    Returns:
        (tuple):  The (n_detector, new_scans, total_samples, group_nodes, memory_bytes)
            of the best configuration.

    """
    log = toast.utils.Logger.get()

    # The output set of observation scans.
    new_scans = list()

    # The number of detectors.  Start with at least enough
    # detectors for the case of group_nodes == 1
    n_detector = n_procs // n_nodes

    # The total samples
    total = 0

    # The process group size
    group_nodes = 0

    # The estimated memory size of the configuration
    memory_bytes = n_procs * per_process_overhead_bytes

    scan_samples = 0
    for isc, sc in enumerate(scans):
        scan_samples += int(sample_rate * (sc.stop - sc.start).total_seconds())
        det_samps = n_detector * scan_samples
        if total == 0:
            # First scan, compute number of detectors
            bytes = None
            while (n_detector < max_n_detector) and (det_samps < max_samples):
                # Increment by whole pixels
                n_detector += 2
                det_samps = n_detector * scan_samples
                group_nodes, bytes = get_minimum_memory_use(
                    n_detector,
                    n_nodes,
                    n_procs,
                    det_samps,
                    scans[: isc + 1],
                    full_pointing,
                )
            if group_nodes == 0:
                msg = f"At maximum detector count ({n_detector}), no compatible "
                msg += f"group size could be found for {n_procs} processes "
                msg += f"across {n_nodes} nodes"
                raise RuntimeError(msg)
            msg = f"Examining first observation, now using {n_detector} detectors"
            log.debug_rank(msg)
            memory_bytes += bytes
            total = det_samps
            new_scans.append(copy.deepcopy(sc))
        else:
            if det_samps > max_samples:
                break
            else:
                group_nodes, bytes = get_minimum_memory_use(
                    n_detector,
                    n_nodes,
                    n_procs,
                    det_samps,
                    scans[: isc + 1],
                    full_pointing,
                )
                if group_nodes == 0:
                    msg = f"For {n_detector} detectors and {det_samps} samples, "
                    msg += f"no compatible group size could be found for "
                    msg += f"{n_procs} processes across {n_nodes} nodes"
                    raise RuntimeError(msg)
                memory_bytes += bytes
                total = det_samps
                new_scans.append(copy.deepcopy(sc))

    return (n_detector, new_scans, total, group_nodes, memory_bytes)


def select_case(
    args,
    n_procs,
    n_nodes,
    avail_node_bytes,
    full_pointing,
    world_comm,
    per_process_overhead_bytes=1024 ** 3,
):
    """
    Selects the most appropriate case size given the memory available and number of
    nodes sets total_samples, n_detector and group_nodes in args.

    One can set `per_process_overhead_bytes` (which defaults to 1GB) to define a number
    of bytes that will be consummed by each process, independently of the number of
    samples, when using case=`auto`.

    When determining the number of detectors and total samples, we start with the first
    observation in the schedule and increase the number of detectors up to the size of
    a nominal focalplane.  Then we add observations to achieve desired number of total
    samples.

    Given the number of detectors and total samples, the group size is chosen to
    minimize total memory use.

    """
    log = toast.utils.Logger.get()

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
            new_scans,
            args.total_samples,
            args.group_nodes,
            memory_used_bytes,
        ) = get_from_samples(
            n_nodes,
            n_procs,
            args.schedule.scans,
            args.max_detector,
            args.sample_rate,
            full_pointing,
            max_samples,
            per_process_overhead_bytes=per_process_overhead_bytes,
        )

        # Update the schedule to use only our subset of scans
        args.schedule.scans = new_scans

        msg = f"Distribution using:\n"
        msg += f"  {args.n_detector} detectors and {len(new_scans)} observations\n"
        msg += f"  {args.total_samples} total samples\n"
        msg += f"  {args.group_nodes} groups of {n_nodes//args.group_nodes} nodes with {n_procs} processes each\n"
        msg += f"  {memory_used_bytes / (1024 ** 3) :0.2f} GB predicted memory use\n"
        msg += f"  ('{args.case}' workflow size)"
        log.info_rank(msg, comm=world_comm)

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
            new_scans,
            args.total_samples,
            args.group_nodes,
            memory_used_bytes,
        ) = maximize_nb_samples(
            n_nodes,
            n_procs,
            args.schedule.scans,
            args.max_detector,
            args.sample_rate,
            full_pointing,
            available_memory_bytes,
            per_process_overhead_bytes=per_process_overhead_bytes,
        )

        # Update the schedule to use only our subset of scans
        args.schedule.scans = new_scans

        msg = f"Distribution using:\n"
        msg += f"  {args.n_detector} detectors and {len(new_scans)} observations\n"
        msg += f"  {args.total_samples} total samples\n"
        msg += f"  {args.group_nodes} groups of {n_nodes//args.group_nodes} nodes with {n_procs} processes each\n"
        msg += f"  {memory_used_bytes / (1024 ** 3) :0.2f} GB predicted memory use "
        msg += f"  ({available_memory_bytes / (1024 ** 3) :0.2f} GB available)\n"
        msg += f"  ('{args.case}' workflow size)"
        log.info_rank(msg, comm=world_comm)


def estimate_memory_overhead(
    n_procs, n_nodes, sky_fraction, nside_solve, nside_final=None
):
    """Estimate bytes of memory used per-process for objects besides timestreams.

    Args:
        n_procs (int):  The number of MPI processes in the job.
        n_nodes (int):  The number of nodes in the job.
        sky_fraction (float):  The fraction of the sky covered by one process.
        nside_solve (int):  The healpix NSIDE value for the solver.
        nside_final (int):  The healpix NSIDE value for the final binning.

    Returns:
        (int):  The bytes used.

    """
    # Start with 1GB for everything else
    base = 1024 ** 3

    # Compute the bytes per pixel.  We have:
    #   hits (int64):  8 bytes
    #   noise weighted map (3 x float64):  24 bytes
    #   condition number map (1 x float64):  8 bytes
    #   diagonal covariance (6 x float64):  48 bytes
    #   solved map (3 x float64):  24 bytes
    bytes_per_pixel = 8 + 24 + 8 + 48 + 24

    n_pixel_solve = sky_fraction * 12 * nside_solve ** 2
    n_pixel_final = 0 if nside_final is None else sky_fraction * 12 * nside_final ** 2

    return base + (n_pixel_solve + n_pixel_final) * bytes_per_pixel


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
                detector_data=trunc_dets, sample_rate=args.sample_rate * u.Hz
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
    tspec = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * numer ** 2 / sig ** 2)
    tspec *= 2000.0

    sig = 100.0
    numer = ell - 500.0
    espec = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * numer ** 2 / sig ** 2)
    espec *= 1.0

    cls = (
        tspec,
        espec,
        np.zeros(3 * nside - 1, dtype=np.float32),
        np.zeros(3 * nside - 1, dtype=np.float32),
    )
    maps = healpy.synfast(
        cls,
        nside,
        pol=True,
        pixwin=False,
        sigma=None,
        new=True,
        fwhm=np.radians(3.0 / 60.0),
        verbose=False,
    )
    healpy.write_map(input_map_path, maps, nest=True, fits_IDL=False, dtype=np.float32)

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
        pointing = job_ops.pointing
        if job_ops.pointing_final.enabled:
            pointing = job_ops.pointing_final
        # creates a map and puts it in args.input_map
        create_input_maps(
            args.input_map, pointing.nside, rank, log, args.print_input_map
        )
        job_ops.scan_map.pixel_dist = job_ops.binner_final.pixel_dist
        job_ops.scan_map.pointing = pointing
        job_ops.scan_map.save_pointing = job_ops.binner_final.full_pointing
        job_ops.scan_map.file = args.input_map
        job_ops.scan_map.apply(data)


def run_mapmaker(job_ops, args, tmpls, data):
    """
    Build up our map-making operation from the pieces- both operators configured from user options and other operators.
    """

    job_ops.binner.noise_model = job_ops.default_model.noise_model
    job_ops.binner_final.noise_model = job_ops.default_model.noise_model

    job_ops.mapmaker.binning = job_ops.binner
    job_ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(
        templates=[tmpls.baselines]
    )
    job_ops.mapmaker.map_binning = job_ops.binner_final
    job_ops.mapmaker.det_data = job_ops.sim_noise.det_data
    job_ops.mapmaker.output_dir = args.out_dir

    # Run the map making
    job_ops.mapmaker.apply(data)


def compute_science_metric(args, runtime, n_nodes, rank, log):
    """
    Computes the science metric and stores it.
    The metric represents the efficiency of the job in a way that is normalized,
    taking the job size into account
    """
    prefactor = 1.0e-3
    kilo_samples = 1.0e-3 * args.total_samples
    sample_factor = 1.2
    det_factor = 2.0
    metric = (
        prefactor
        * args.n_detector ** det_factor
        * kilo_samples ** sample_factor
        / (n_nodes * runtime)
    )
    if rank == 0:
        msg = f"Science Metric: {prefactor:0.1e} * ({args.n_detector:d}**{det_factor:0.2f}) * ({kilo_samples:0.3e}**{sample_factor:0.3f}) / ({runtime:0.1f} * {n_nodes}) = {metric:0.2f}"
        log.info("")
        log.info(msg)
        log.info("")
        with open(os.path.join(args.out_dir, "log"), "a") as f:
            f.write(msg)
            f.write("\n\n")
