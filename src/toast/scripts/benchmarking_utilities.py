# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
Functions common to all benchmarking scripts.

total_sample = num_obs * obs_minutes * sample_rate * n_detector
"""

import math
import healpy
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import toast
from toast.job import job_size, get_node_mem
from toast.instrument_sim import fake_hexagon_focalplane


def get_mpi_settings(args, log, env):
    """
    Getting the MPI settings
    taking the dry_run parameter into account
    """
    # gets actual MPI information
    world_comm, procs, rank = toast.get_world()
    log.info_rank0(f"TOAST version = {env.version()}", world_comm)
    log.info_rank0(
        f"Using a maximum of {env.max_threads()} threads per process", world_comm
    )
    if world_comm is None:
        log.info_rank0(
            f"Running serially with one process at {str(datetime.now())}", world_comm
        )
    else:
        log.info_rank0(
            f"Running with {procs} processes at {str(datetime.now())}", world_comm
        )

    # is this a dry run that does not use MPI
    if args.dry_run is not None:
        procs, procs_per_node = args.dry_run.split(",")
        procs = int(procs)
        procs_per_node = int(procs_per_node)
        log.info_rank0(
            f"DRY RUN simulating {procs} total processes with {procs_per_node} per node",
            world_comm,
        )
        # We are simulating the distribution
        min_avail, max_avail = get_node_mem(world_comm, 0)
        avail_node_bytes = max_avail
    else:
        # Get information about the actual job size
        (n_node, procs_per_node, min_avail, max_avail) = job_size(world_comm)
        avail_node_bytes = max_avail

    # sets per node memory
    log.info_rank0(
        f"Minimum detected per-node memory available is {avail_node_bytes / (1024 ** 3) :0.2f} GB",
        world_comm,
    )
    if args.node_mem_gb is not None:
        avail_node_bytes = int((1024 ** 3) * args.node_mem_gb)
        log.info_rank0(
            f"Setting per-node available memory to {avail_node_bytes / (1024 ** 3) :0.2f} GB as requested",
            world_comm,
        )

    # computes the total number of nodes
    n_nodes = procs // procs_per_node
    log.info_rank0(f"Job has {n_nodes} total nodes", world_comm)

    return world_comm, procs, rank, n_nodes, avail_node_bytes


def get_minimum_memory_use(args, n_nodes, n_procs, total_samples, full_pointing):
    """
    Given a number of samples and some problems parameters,
    returns (group_nodes, n_detector, memory_used_bytes)
    such that memory_used_bytes is minimized.
    """
    # memory usage of the samples
    detector_timestream_cost = (1 + 4) if full_pointing else 1
    det_bytes_per_sample = 2 * (  # At most 2 detector data copies.
        8  # 64 bit float / ints used
        * detector_timestream_cost  # detector timestream  # pixel index and 3 IQU weights
        + 1  # one byte per sample for flags
    )
    common_bytes_per_sample = (
        8 * (4)  # 64 bit floats  # One quaternion per sample
        + 1  # one byte per sample for common flag
    )
    # Minimum time span (one day)
    min_time_samples = int(24 * 3600 * args.sample_rate)
    # group_nodes is the number of nodes in each group, it should be a divisor of n_nodes
    group_nodes_best = 1
    n_detector_best = 1
    num_obs_best = 1
    memory_used_bytes_best = np.inf
    # what is the minimum memory we can use for that number of samples?
    for group_nodes in range(1, n_nodes + 1):
        if n_nodes % group_nodes == 0:
            # For the minimum time span, scale up the number of detectors to reach the requested total sample size.
            n_detector = np.clip(
                total_samples // min_time_samples,
                a_min=n_procs // group_nodes,
                a_max=args.max_detector,
            )
            num_obs = max(
                1, (args.obs_minutes * args.sample_rate * n_detector) // total_samples
            )
            bytes_per_samp = (
                n_detector * det_bytes_per_sample
                + group_nodes * common_bytes_per_sample
            )
            memory_used_bytes = bytes_per_samp * (total_samples // n_detector)
            # needs to fit in memory and be able to split observations evenly between groups
            if (memory_used_bytes < memory_used_bytes_best) and (
                num_obs % group_nodes == 0
            ):
                group_nodes_best = group_nodes
                n_detector_best = n_detector
                num_obs_best = num_obs
                memory_used_bytes_best = memory_used_bytes
    # returns the group_nodes and n_detector that minimize memory usage
    return (group_nodes_best, n_detector_best, num_obs_best, memory_used_bytes_best)


def select_case(
    args, n_procs, n_nodes, avail_node_bytes, full_pointing, world_comm, log
):
    """
    Selects the most appropriate case size given the memory available and number of nodes
    sets total_samples and n_detector in args
    """
    # computes the memory that is currently available
    available_memory_bytes = n_nodes * avail_node_bytes

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

    if args.case != "auto":
        # force use the case size suggested by the user
        args.total_samples = cases_samples[args.case]
        # finds the parameters that minimize memory use
        (group_nodes, n_detector, num_obs, memory_used_bytes) = get_minimum_memory_use(
            args, n_nodes, n_procs, args.total_samples, full_pointing
        )
        args.n_detector = n_detector
        args.num_obs = num_obs
        log.info_rank0(
            f"Distribution using {args.total_samples} total samples spread over {group_nodes} groups of {n_nodes//group_nodes} nodes that have {n_procs} processors each ('{args.case}' workflow size)",
            world_comm,
        )
        log.info_rank0(
            f"Using {num_obs} observations produced at {args.obs_minutes} observation/minute.",
            world_comm,
        )
        if (memory_used_bytes >= available_memory_bytes) and (
            (world_comm is None) or (world_comm.rank == 0)
        ):
            log.warning(
                f"The selected case, '{args.case}' might not fit in memory (we predict a usage of about {memory_used_bytes / (1024 ** 3) :0.2f} GB)."
            )
    else:
        # tries the workflow sizes from largest to smalest, until we find one that fits
        for (case, total_samples) in cases_samples.items():
            # finds the parameters that minimize memory use
            (
                group_nodes,
                n_detector,
                num_obs,
                memory_used_bytes,
            ) = get_minimum_memory_use(
                args, n_nodes, n_procs, total_samples, full_pointing
            )
            # log.info_rank0(f"DEBUG: Distribution using {total_samples} total samples spread over {group_nodes} groups of {n_nodes//group_nodes} nodes that have {n_procs} processors each ('{case}' workflow size), (we predict a usage of about {memory_used_bytes / (1024 ** 3) :0.2f} GB)", world_comm)
            # if the parameters fit in memory, we are done
            if memory_used_bytes < available_memory_bytes:
                args.case = case
                args.total_samples = total_samples
                args.n_detector = n_detector
                args.num_obs = num_obs
                log.info_rank0(
                    f"Distribution using {total_samples} total samples spread over {group_nodes} groups of {n_nodes//group_nodes} nodes that have {n_procs} processors each ('{case}' workflow size)",
                    world_comm,
                )
                log.info_rank0(
                    f"Using {num_obs} observations produced at {args.obs_minutes} observation/minute (we predict a usage of about {memory_used_bytes / (1024 ** 3) :0.2f} GB which should be below the available {available_memory_bytes / (1024 ** 3) :0.2f} GB).",
                    world_comm,
                )
                return
        raise Exception("Error: Unable to fit a case size in memory!")


def make_focalplane(args, world_comm, log):
    """
    Creates a fake focalplane
    """
    # computes the number of pixels to be used
    ring = math.ceil(math.sqrt((args.n_detector - 2) / 6)) if args.n_detector > 2 else 0
    n_pixel = 1 + 3 * ring * (ring + 1)
    log.info_rank0(f"Using {n_pixel} hexagon-packed pixels.", world_comm)
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
    if world_comm is not None:
        focalplane = world_comm.bcast(focalplane, root=0)
    return focalplane


def create_input_maps(input_map_path, nside, rank, log):
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
    healpy.mollview(maps[0])
    plt.savefig(f"{input_map_path}_fake-T.png")
    plt.close()

    healpy.mollview(maps[1])
    plt.savefig(f"{input_map_path}_fake-E.png")
    plt.close()


def scan_map(args, rank, ops, data, log):
    """
    Simulate sky signal from a map.
    We scan the sky with the "final" pointing model if that is different from the solver pointing model.
    """
    pix_dist = toast.ops.BuildPixelDistribution()
    if ops.binner_final.enabled and ops.pointing_final.enabled:
        pix_dist.pixel_dist = ops.binner_final.pixel_dist
        pix_dist.pointing = ops.pointing_final
        pix_dist.shared_flags = ops.binner_final.shared_flags
        pix_dist.shared_flag_mask = ops.binner_final.shared_flag_mask
        pix_dist.save_pointing = ops.binner_final.full_pointing
    else:
        pix_dist.pixel_dist = ops.binner.pixel_dist
        pix_dist.pointing = ops.pointing
        pix_dist.shared_flags = ops.binner.shared_flags
        pix_dist.shared_flag_mask = ops.binner.shared_flag_mask
        pix_dist.save_pointing = ops.binner.full_pointing
    pix_dist.apply(data)

    # creates a map and puts it in args.input_map
    create_input_maps(args.input_map, ops.pointing.nside, rank, log)

    # adds the scan map operator
    scan_map = toast.ops.ScanHealpix(
        pixel_dist=pix_dist.pixel_dist, pointing=pix_dist.pointing, file=args.input_map
    )
    scan_map.apply(data)


def run_mapmaker(ops, args, tmpls, data):
    """
    Build up our map-making operation from the pieces- both operators configured from user options and other operators.
    """
    ops.binner.pointing = ops.pointing
    ops.binner.noise_model = ops.default_model.noise_model

    final_bin = None
    if ops.binner_final.enabled:
        final_bin = ops.binner_final
        if ops.pointing_final.enabled:
            final_bin.pointing = ops.pointing_final
        else:
            final_bin.pointing = ops.pointing
        final_bin.noise_model = ops.default_model.noise_model

    # A simple binned map will be made if an empty list of templates is passed to the mapmaker.
    tlist = list()
    if tmpls.baselines.enabled:
        tlist.append(tmpls.baselines)
    tmatrix = toast.ops.TemplateMatrix(templates=tlist)

    ops.mapmaker.binning = ops.binner
    ops.mapmaker.template_matrix = tmatrix
    ops.mapmaker.map_binning = final_bin
    ops.mapmaker.det_data = ops.sim_noise.det_data

    # Run the map making
    ops.mapmaker.apply(data)

    # Write the outputs
    for prod in ["map", "hits", "cov", "rcond"]:
        dkey = f"{ops.mapmaker.name}_{prod}"
        file = os.path.join(args.out_dir, f"{dkey}.fits")
        toast.pixels_io.write_healpix_fits(data[dkey], file, nest=ops.pointing.nest)


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
