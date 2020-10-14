#!/usr/bin/env python3

# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script dynamically builds a workflow tailored to the size of the communicator.
"""

# This is the maximum random time in seconds for every process to sleep before starting.
# This should be long enough to avoid filesystem contention when loading shared
# libraries and python modules, but short enough that the scheduler does not think that
# MPI has hung and kills the job.
startup_delay = 2.0

# These are builtin modules, hopefully fast to load.
import random
import time

wait = random.uniform(0.0, startup_delay)
time.sleep(wait)

from toast.mpi import MPI

# Now import the remaining modules
import os
import sys
import re
import copy
import argparse
import traceback
import pickle

from datetime import datetime

import psutil

import numpy as np

import healpy as hp

from toast.mpi import get_world, Comm

from toast.dist import distribute_uniform, Data, distribute_discrete

from toast.utils import Logger, Environment, memreport

from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing

from toast.todmap import TODGround

from toast import pipeline_tools

from toast import rng

from toast.pipeline_tools import Focalplane

from toast.tod.sim_focalplane import hex_pol_angles_qu, hex_layout, hex_nring

from toast.schedule import run_scheduler


# This function is part of the package in TOAST 3.0.  Remove this here when porting.


def fake_hexagon_focalplane(
    n_pix=7,
    width_deg=5.0,
    samplerate=1.0,
    epsilon=0.0,
    net=1.0,
    fmin=0.0,
    alpha=1.0,
    fknee=0.05,
):
    pol_A = hex_pol_angles_qu(n_pix, offset=0.0)
    pol_B = hex_pol_angles_qu(n_pix, offset=90.0)
    quat_A = hex_layout(n_pix, width_deg, "D", "A", pol_A)
    quat_B = hex_layout(n_pix, width_deg, "D", "B", pol_B)

    det_data = dict(quat_A)
    det_data.update(quat_B)

    nrings = hex_nring(n_pix)
    detfwhm = 0.5 * 60.0 * width_deg / (2 * nrings - 1)

    for det in det_data.keys():
        det_data[det]["pol_leakage"] = epsilon
        det_data[det]["fmin"] = fmin
        det_data[det]["fknee"] = fknee
        det_data[det]["alpha"] = alpha
        det_data[det]["NET"] = net
        det_data[det]["fwhm_arcmin"] = detfwhm
        det_data[det]["fsample"] = samplerate

    return Focalplane(detector_data=det_data, sample_rate=samplerate)


def get_node_mem(mpicomm, node_rank):
    avail = 2 ** 62
    if node_rank == 0:
        vmem = psutil.virtual_memory()._asdict()
        avail = vmem["available"]
    if mpicomm is not None:
        avail = mpicomm.allreduce(avail, op=MPI.MIN)
    return int(avail)


def sample_distribution(
    rank, procs_per_node, bytes_per_node, total_samples, sample_rate
):
    # For this benchmark, we start by ramping up to a realistic number of detectors for
    # one day of data. Then we extend the timespan to achieve the desired number of
    # samples.

    log = Logger.get()

    # Hex-packed 127 pixels (6 rings) times two dets per pixel.
    # max_detector = 254

    # Hex-packed 1027 pixels (18 rings) times two dets per pixel.
    max_detector = 2054

    # Minimum time span (one day)
    min_time_samples = int(24 * 3600 * sample_rate)

    # For the minimum time span, scale up the number of detectors to reach the
    # requested total sample size.

    n_detector = 1
    test_samples = n_detector * min_time_samples

    while test_samples < total_samples and n_detector < max_detector:
        n_detector += 1
        test_samples = n_detector * min_time_samples

    if rank == 0:
        log.debug(
            "  Dist total = {}, using {} detectors at min time samples = {}".format(
                total_samples, n_detector, min_time_samples
            )
        )

    # For this number of detectors, determine the group size needed to fit the
    # minimum number of samples in memory.  In practice, one day will actually be
    # split up into multiple observations.  However, sizing the groups this way ensures
    # that each group will have multiple observations and improve the load balancing.

    det_bytes_per_sample = 2 * (  # At most 2 detector data copies.
        8  # 64 bit float / ints used
        * (1 + 4)  # detector timestream  # pixel index and 3 IQU weights
        + 1  # one byte per sample for flags
    )

    common_bytes_per_sample = (
        8 * (4)  # 64 bit floats  # One quaternion per sample
        + 1  # one byte per sample for common flag
    )

    group_nodes = 0
    group_mem = 0.0

    # This just ensures we go through the loop once.
    min_time_mem = 1.0

    while group_mem < min_time_mem:
        group_nodes += 1
        group_procs = group_nodes * procs_per_node
        group_mem = group_nodes * bytes_per_node

        # NOTE:  change this when moving to toast-3, since common data is in shared mem.
        # So the prefactor should be nodes per group, not group_procs.
        bytes_per_samp = (
            n_detector * det_bytes_per_sample + group_procs * common_bytes_per_sample
        )
        # bytes_per_samp = (
        #     n_detector * det_bytes_per_sample + group_nodes * common_bytes_per_sample
        # )
        min_time_mem = min_time_samples * bytes_per_samp
        if rank == 0:
            log.verbose(
                "  Dist testing {} group nodes, {} proc/node, group mem = {}, comparing to minimum = {} ({} samp * {} bytes/samp)".format(
                    group_nodes,
                    procs_per_node,
                    group_mem,
                    min_time_mem,
                    min_time_samples,
                    bytes_per_samp,
                )
            )

    if rank == 0:
        log.debug("  Dist selecting {} nodes per group".format(group_nodes))

    # Now set the number of groups to get the target number of total samples.

    group_time_samples = min_time_samples
    group_samples = n_detector * group_time_samples

    n_group = 1 + (total_samples // group_samples)

    time_samples = n_group * group_time_samples

    if rank == 0:
        log.debug(
            "  Dist using {} groups, each with {} / {} (time / total) samples".format(
                n_group, group_time_samples, group_samples
            )
        )
        log.debug("  Dist using {} total samples".format(n_detector * time_samples))

    return (
        n_detector,
        time_samples,
        group_procs,
        group_nodes,
        n_group,
        group_time_samples,
    )


def job_size(mpicomm):
    log = Logger.get()

    procs_per_node = 1
    node_rank = 0
    nodecomm = None
    rank = 0
    procs = 1

    if mpicomm is not None:
        rank = mpicomm.rank
        procs = mpicomm.size
        nodecomm = mpicomm.Split_type(MPI.COMM_TYPE_SHARED, 0)
        node_rank = nodecomm.rank
        procs_per_node = nodecomm.size
        min_per_node = mpicomm.allreduce(procs_per_node, op=MPI.MIN)
        max_per_node = mpicomm.allreduce(procs_per_node, op=MPI.MAX)
        if min_per_node != max_per_node:
            raise RuntimeError("Nodes have inconsistent numbers of MPI ranks")

    # One process on each node gets available RAM and communicates it
    avail = get_node_mem(mpicomm, node_rank)

    n_node = procs // procs_per_node

    if rank == 0:
        log.info(
            "Job running on {} nodes each with {} processes ({} total)".format(
                n_node, procs_per_node, procs
            )
        )
    return (procs_per_node, avail)


def job_config(mpicomm, cases):
    env = Environment.get()
    log = Logger.get()

    class args:
        debug = False
        # TOD Ground options
        el_mod_step_deg = 0.0
        el_mod_rate_hz = 0.0
        el_mod_amplitude_deg = 1.0
        el_mod_sine = False
        el_nod_deg = False
        el_nod_every_scan = False
        start_with_el_nod = False
        end_with_el_nod = False
        scan_rate = 1.0
        scan_rate_el = 0.0
        scan_accel = 1.0
        scan_accel_el = 0.0
        scan_cosecant_modulate = False
        sun_angle_min = 30.0
        schedule = None  # required
        weather = "SIM"
        timezone = 0
        sample_rate = 100.0
        coord = "C"
        split_schedule = None
        sort_schedule = False
        hwp_rpm = 10.0
        hwp_step_deg = None
        hwp_step_time_s = None
        elevation_noise_a = 0.0
        elevation_noise_b = 0.0
        freq = "150"
        do_daymaps = False
        do_seasonmaps = False
        # Pointing options
        nside = 1024
        nside_submap = 16
        single_precision_pointing = False
        common_flag_mask = 1
        # Polyfilter options
        apply_polyfilter = False
        poly_order = 0
        # Ground filter options
        apply_groundfilter = False
        ground_order = 0
        # Atmosphere options
        simulate_atmosphere = False
        simulate_coarse_atmosphere = False
        focalplane_radius_deg = None
        atm_verbosity = 0
        atm_lmin_center = 0.01
        atm_lmin_sigma = 0.001
        atm_lmax_center = 10.0
        atm_lmax_sigma = 10.0
        atm_gain = 2.0e-5
        atm_gain_coarse = 8.0e-5
        atm_zatm = 40000.0
        atm_zmax = 200.0
        atm_xstep = 10.0
        atm_ystep = 10.0
        atm_zstep = 10.0
        atm_nelem_sim_max = 10000
        atm_wind_dist = 3000.0
        atm_z0_center = 2000.0
        atm_z0_sigma = 0.0
        atm_T0_center = 280.0
        atm_T0_sigma = 10.0
        atm_cache = None
        atm_apply_flags = False
        # Noise simulation options
        simulate_noise = False
        # Gain scrambler
        apply_gainscrambler = False
        gain_sigma = 0.01
        # Map maker
        mapmaker_prefix = "toast"
        mapmaker_mask = None
        mapmaker_weightmap = None
        mapmaker_iter_max = 20
        mapmaker_precond_width = 100
        mapmaker_prefilter_order = None
        mapmaker_baseline_length = 200.0
        mapmaker_noisefilter = False
        mapmaker_fourier2D_order = None
        mapmaker_fourier2D_subharmonics = None
        write_hits = True
        write_binmap = True
        write_wcov = False
        write_wcov_inv = False
        zip_maps = False
        # Monte Carlo
        MC_start = 0
        MC_count = 1
        # Sky signal
        input_map = None
        simulate_sky = True
        # Input dir
        auxdir = "toast_inputs"
        # Output
        outdir = "toast"
        tidas = None
        spt3g = None

    parser = argparse.ArgumentParser(
        description="Run a TOAST workflow scaled appropriately to the MPI communicator size and available memory.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--node_mem_gb",
        required=False,
        default=None,
        type=float,
        help="Use this much memory per node in GB",
    )

    parser.add_argument(
        "--dry_run",
        required=False,
        default=None,
        type=str,
        help="Comma-separated total_procs,node_procs to simulate.",
    )

    parser.parse_args(namespace=args)

    procs = 1
    rank = 0
    if mpicomm is not None:
        procs = mpicomm.size
        rank = mpicomm.rank

    avail_node_bytes = None
    procs_per_node = None

    if args.dry_run is not None:
        dryrun_total, dryrun_node = args.dry_run.split(",")
        dryrun_total = int(dryrun_total)
        dryrun_node = int(dryrun_node)
        if rank == 0:
            log.info(
                "DRY RUN simulating {} total processes with {} per node".format(
                    dryrun_total, dryrun_node
                )
            )
        procs_per_node = dryrun_node
        procs = dryrun_total
        # We are simulating the distribution
        avail_node_bytes = get_node_mem(mpicomm, 0)

    else:
        # Get information about the actual job size
        procs_per_node, avail_node_bytes = job_size(mpicomm)

    if rank == 0:
        log.info(
            "Minimum detected per-node memory available is {:0.2f} GB".format(
                avail_node_bytes / (1024 ** 3)
            )
        )

    if args.node_mem_gb is not None:
        avail_node_bytes = int((1024 ** 3) * args.node_mem_gb)
        if rank == 0:
            log.info(
                "Setting per-node available memory to {:0.2f} GB as requested".format(
                    avail_node_bytes / (1024 ** 3)
                )
            )

    # Based on the total number of processes and count per node, choose the number of
    # nodes in each observation and a focalplane such that every process has >= 4
    # detectors.

    n_nodes = procs // procs_per_node
    if rank == 0:
        log.info("Job has {} total nodes".format(n_nodes))

    if rank == 0:
        log.info("Examining {} possible cases to run:".format(len(cases)))

    selected_case = None
    selected_nodes = None
    n_detector = None
    time_samples = None
    group_procs = None
    group_nodes = None
    n_group = None
    group_time_samples = None

    for case_name, case_samples in cases.items():
        (
            case_n_detector,
            case_time_samples,
            case_group_procs,
            case_group_nodes,
            case_n_group,
            case_group_time_samples,
        ) = sample_distribution(
            rank, procs_per_node, avail_node_bytes, case_samples, args.sample_rate
        )

        case_min_nodes = case_n_group * case_group_nodes
        if rank == 0:
            log.info(
                "  {:8s}: requires {:d} nodes for {} MPI ranks and {:0.1f}GB per node".format(
                    case_name,
                    case_min_nodes,
                    procs_per_node,
                    avail_node_bytes / (1024 ** 3),
                )
            )

        if selected_nodes is None:
            if case_min_nodes <= n_nodes:
                # First case that fits in our job
                selected_case = case_name
                selected_nodes = case_min_nodes
                n_detector = case_n_detector
                time_samples = case_time_samples
                group_procs = case_group_procs
                group_nodes = case_group_nodes
                n_group = case_n_group
                group_time_samples = case_group_time_samples
        else:
            if (case_min_nodes <= n_nodes) and (case_min_nodes >= selected_nodes):
                # This case fits in our job and is larger than the current one
                selected_case = case_name
                selected_nodes = case_min_nodes
                n_detector = case_n_detector
                time_samples = case_time_samples
                group_procs = case_group_procs
                group_nodes = case_group_nodes
                n_group = case_n_group
                group_time_samples = case_group_time_samples

    if selected_case is None:
        msg = (
            "None of the available cases fit into aggregate memory.  Use a larger job."
        )
        if rank == 0:
            log.error(msg)
        raise RuntimeError(msg)
    else:
        if rank == 0:
            log.info("Selected case '{}'".format(selected_case))

    if rank == 0:
        log.info("Using groups of {} nodes".format(group_nodes))

    # Adjust number of groups

    if n_nodes % group_nodes != 0:
        msg = "Current number of nodes ({}) is not divisible by the required group size ({})".format(
            n_nodes, group_nodes
        )
        if rank == 0:
            log.error(msg)
        raise RuntimeError(msg)

    n_group = n_nodes // group_nodes
    group_time_samples = 1 + time_samples // n_group

    group_seconds = group_time_samples / args.sample_rate

    if args.simulate_atmosphere and args.weather is None:
        raise RuntimeError("Cannot simulate atmosphere without a TOAST weather file")

    comm = None
    if mpicomm is None or args.dry_run is not None:
        comm = Comm(world=None)
    else:
        comm = Comm(world=mpicomm, groupsize=group_procs)

    jobdate = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    args.outdir += "_{:06d}_grp-{:04d}p-{:02d}n_{}".format(
        procs, group_procs, group_nodes, jobdate
    )
    args.auxdir = os.path.join(args.outdir, "inputs")

    if rank == 0:
        os.makedirs(args.outdir)
        os.makedirs(args.auxdir, exist_ok=True)

    if rank == 0:
        with open(os.path.join(args.outdir, "log"), "w") as f:
            f.write("Running at {}\n".format(jobdate))
            f.write("TOAST version = {}\n".format(env.version()))
            f.write("TOAST max threads = {}\n".format(env.max_threads()))
            f.write("MPI Processes = {}\n".format(procs))
            f.write("MPI Processes per node = {}\n".format(procs_per_node))
            f.write(
                "Memory per node = {:0.2f} GB\n".format(avail_node_bytes / (1024 ** 3))
            )
            f.write("Number of groups = {}\n".format(n_group))
            f.write("Group nodes = {}\n".format(group_nodes))
            f.write("Group MPI Processes = {}\n".format(group_procs))
            f.write("Case selected = {}\n".format(selected_case))
            f.write("Case number of detectors = {}\n".format(n_detector))
            f.write(
                "Case total samples = {}\n".format(
                    n_group * group_time_samples * n_detector
                )
            )
            f.write(
                "Case samples per group = {}\n".format(group_time_samples * n_detector)
            )
            f.write("Case data seconds per group = {}\n".format(group_seconds))
            f.write("Parameters:\n")
            for k, v in vars(args).items():
                if re.match(r"_.*", k) is None:
                    f.write("  {} = {}\n".format(k, v))

    args.schedule = os.path.join(args.auxdir, "schedule.txt")
    args.input_map = os.path.join(args.auxdir, "cmb.fits")

    return args, comm, n_nodes, n_detector, selected_case, group_seconds, n_group


def create_schedules(args, max_ces_seconds, days):
    opts = [
        "--site-lat",
        str(-22.958064),
        "--site-lon",
        str(-67.786222),
        "--site-alt",
        str(5200.0),
        "--site-name",
        "ATACAMA",
        "--telescope",
        "atacama_telescope",
        "--patch-coord",
        "C",
        "--el-min-deg",
        str(30.0),
        "--el-max-deg",
        str(80.0),
        "--sun-el-max-deg",
        str(90.0),
        "--sun-avoidance-angle-deg",
        str(30.0),
        "--moon-avoidance-angle-deg",
        str(10.0),
        "--start",
        "2021-06-01 00:00:00",
        "--gap-s",
        str(600.0),
        "--gap-small-s",
        str(0.0),
        "--fp-radius-deg",
        str(0.0),
        "--patch",
        "BICEP,1,-10,-55,10,-58",
        "--ces-max-time-s",
        str(max_ces_seconds),
        "--operational-days",
        str(days),
    ]

    if not os.path.isfile(args.schedule):
        log = Logger.get()
        log.info("Generating input schedule file {}:".format(args.schedule))

        opts.extend(["--out", args.schedule])
        run_scheduler(opts=opts)


def create_input_maps(args):
    if not os.path.isfile(args.input_map):
        log = Logger.get()
        log.info("Generating input map {}".format(args.input_map))

        # This is *completely* fake- just to have something on the sky besides zeros.

        ell = np.arange(3 * args.nside - 1, dtype=np.float64)

        sig = 50.0
        numer = ell - 30.0
        tspec = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * numer ** 2 / sig ** 2
        )
        tspec *= 2000.0

        sig = 100.0
        numer = ell - 500.0
        espec = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * numer ** 2 / sig ** 2
        )
        espec *= 1.0

        cls = (
            tspec,
            espec,
            np.zeros(3 * args.nside - 1, dtype=np.float32),
            np.zeros(3 * args.nside - 1, dtype=np.float32),
        )
        maps = hp.synfast(
            cls,
            args.nside,
            pol=True,
            pixwin=False,
            sigma=None,
            new=True,
            fwhm=np.radians(3.0 / 60.0),
            verbose=False,
        )
        # for m in maps:
        #     hp.reorder(m, inp="RING", out="NEST")
        hp.write_map(args.input_map, maps, nest=True, fits_IDL=False, dtype=np.float32)

        import matplotlib.pyplot as plt

        hp.mollview(maps[0])
        plt.savefig("{}_fake-T.png".format(args.input_map))
        plt.close()

        hp.mollview(maps[1])
        plt.savefig("{}_fake-E.png".format(args.input_map))
        plt.close()


@function_timer
def create_focalplanes(args, comm, schedules, n_detector):
    """Attach a focalplane to each of the schedules.

    Args:
        schedules (list) :  List of Schedule instances.
            Each schedule has two members, telescope
            and ceslist, a list of CES objects.
    Returns:
        detweights (dict) : Inverse variance noise weights for every
            detector across all focal planes. In [K_CMB^-2].
            They can be used to bin the TOD.
    """

    # Load focalplane information

    focalplanes = []
    if comm.world_rank == 0:
        # First create a fake hexagonal focalplane
        n_pixel = 1
        ring = 1
        while 2 * n_pixel < n_detector:
            n_pixel += 6 * ring
            ring += 1

        fake = fake_hexagon_focalplane(
            n_pix=n_pixel,
            width_deg=10.0,
            samplerate=100.0,
            epsilon=0.0,
            net=10.0,
            fmin=0.0,
            alpha=1.0,
            fknee=0.05,
        )

        # Now truncate the detectors to the desired count.
        newdat = dict()
        off = 0
        for k, v in fake.detector_data.items():
            newdat[k] = v
            off += 1
            if off >= n_detector:
                break
        fake.detector_data = newdat
        fake.reset_properties()
        focalplanes.append(fake)

    if comm.comm_world is not None:
        focalplanes = comm.comm_world.bcast(focalplanes)

    if len(focalplanes) == 1 and len(schedules) > 1:
        focalplanes *= len(schedules)

    # Append a focal plane and telescope to each entry in the schedules
    # list and assemble a detector weight dictionary that represents all
    # detectors in all focalplanes
    detweights = {}
    for schedule, focalplane in zip(schedules, focalplanes):
        schedule.telescope.focalplane = focalplane
        detweights.update(schedule.telescope.focalplane.detweights)

    return detweights


@function_timer
def create_observation(args, comm, telescope, ces, verbose=True):
    """Create a TOAST observation.

    Create an observation for the CES scan

    Args:
        args :  argparse arguments
        comm :  TOAST communicator
        ces (CES) :  One constant elevation scan

    """
    focalplane = telescope.focalplane
    site = telescope.site
    weather = site.weather
    noise = focalplane.noise
    totsamples = int((ces.stop_time - ces.start_time) * args.sample_rate)

    # create the TOD for this observation

    if comm.comm_group is not None:
        ndetrank = comm.comm_group.size
    else:
        ndetrank = 1

    if args.el_nod_deg and (ces.subscan == 0 or args.el_nod_every_scan):
        el_nod = args.el_nod_deg
    else:
        el_nod = None

    try:
        tod = TODGround(
            comm.comm_group,
            focalplane.detquats,
            totsamples,
            detranks=ndetrank,
            boresight_angle=ces.boresight_angle,
            firsttime=ces.start_time,
            rate=args.sample_rate,
            site_lon=site.lon,
            site_lat=site.lat,
            site_alt=site.alt,
            azmin=ces.azmin,
            azmax=ces.azmax,
            el=ces.el,
            el_nod=el_nod,
            start_with_elnod=args.start_with_el_nod,
            end_with_elnod=args.end_with_el_nod,
            el_mod_step=args.el_mod_step_deg,
            el_mod_rate=args.el_mod_rate_hz,
            el_mod_amplitude=args.el_mod_amplitude_deg,
            el_mod_sine=args.el_mod_sine,
            scanrate=args.scan_rate,
            scanrate_el=args.scan_rate_el,
            scan_accel=args.scan_accel,
            scan_accel_el=args.scan_accel_el,
            cosecant_modulation=args.scan_cosecant_modulate,
            CES_start=None,
            CES_stop=None,
            sun_angle_min=args.sun_angle_min,
            coord=args.coord,
            sampsizes=None,
            report_timing=args.debug,
            hwprpm=args.hwp_rpm,
            hwpstep=args.hwp_step_deg,
            hwpsteptime=args.hwp_step_time_s,
        )
    except RuntimeError as e:
        raise RuntimeError(
            'Failed to create TOD for {}-{}-{}: "{}"'
            "".format(ces.name, ces.scan, ces.subscan, e)
        )

    # Create the observation

    obs = {}
    obs["name"] = "CES-{}-{}-{}-{}-{}".format(
        site.name, telescope.name, ces.name, ces.scan, ces.subscan
    )
    obs["tod"] = tod
    obs["baselines"] = None
    obs["noise"] = copy.deepcopy(noise)
    obs["id"] = int(ces.mjdstart * 10000)
    obs["intervals"] = tod.subscans
    obs["site"] = site
    obs["site_name"] = site.name
    obs["site_id"] = site.id
    obs["altitude"] = site.alt
    obs["weather"] = site.weather
    obs["telescope"] = telescope
    obs["telescope_name"] = telescope.name
    obs["telescope_id"] = telescope.id
    obs["focalplane"] = telescope.focalplane.detector_data
    obs["fpradius"] = telescope.focalplane.radius
    obs["start_time"] = ces.start_time
    obs["season"] = ces.season
    obs["date"] = ces.start_date
    obs["MJD"] = ces.mjdstart
    obs["rising"] = ces.rising
    obs["mindist_sun"] = ces.mindist_sun
    obs["mindist_moon"] = ces.mindist_moon
    obs["el_sun"] = ces.el_sun
    return obs


@function_timer
def create_observations(args, comm, schedules):
    """Create and distribute TOAST observations for every CES in
    schedules.

    Args:
        schedules (iterable) :  a list of Schedule objects.
    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    data = Data(comm)

    # Loop over the schedules, distributing each schedule evenly across
    # the process groups.  For now, we'll assume that each schedule has
    # the same number of operational days and the number of process groups
    # matches the number of operational days.  Relaxing these constraints
    # will cause the season break to occur on different process groups
    # for different schedules and prevent splitting the communicator.

    total_samples = 0
    group_samples = 0
    for schedule in schedules:

        telescope = schedule.telescope
        all_ces = schedule.ceslist
        nces = len(all_ces)

        breaks = pipeline_tools.get_breaks(comm, all_ces, args)

        ces_weights = [x.stop_time - x.start_time for x in all_ces]
        groupdist = distribute_discrete(ces_weights, comm.ngroups, breaks=breaks)

        # groupdist = distribute_uniform(nces, comm.ngroups, breaks=breaks)
        group_firstobs = groupdist[comm.group][0]
        group_numobs = groupdist[comm.group][1]

        for ices in range(group_firstobs, group_firstobs + group_numobs):
            obs = create_observation(args, comm, telescope, all_ces[ices])
            group_samples += obs["tod"].total_samples
            data.obs.append(obs)

    if comm.comm_rank is not None:
        if comm.comm_group.rank == 0:
            total_samples = comm.comm_rank.allreduce(group_samples, op=MPI.SUM)
        total_samples = comm.comm_group.bcast(total_samples, root=0)
    if comm.comm_world is None or comm.comm_group.rank == 0:
        log.info("Group # {:4} has {} observations.".format(comm.group, len(data.obs)))

    if len(data.obs) == 0:
        raise RuntimeError(
            "Too many tasks. Every MPI task must "
            "be assigned to at least one observation."
        )

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Simulated scans")

    # Split the data object for each telescope for separate mapmaking.
    # We could also split by site.

    if len(schedules) > 1:
        telescope_data = data.split("telescope_name")
        if len(telescope_data) == 1:
            # Only one telescope available
            telescope_data = []
    else:
        telescope_data = []
    telescope_data.insert(0, ("all", data))
    return data, telescope_data, total_samples


def setup_sigcopy(args):
    """Determine if an extra copy of the atmospheric signal is needed.

    When we simulate multichroic focal planes, the frequency-independent
    part of the atmospheric noise is simulated first and then the
    frequency scaling is applied to a copy of the atmospheric noise.
    """
    if len(args.freq.split(",")) == 1:
        totalname = "total"
        totalname_freq = "total"
    else:
        totalname = "total"
        totalname_freq = "total_freq"

    return totalname, totalname_freq


@function_timer
def setup_output(args, comm, mc, freq):
    outpath = "{}/{:03}/{:03}".format(args.outdir, mc, int(freq))
    if comm.world_rank == 0:
        os.makedirs(outpath, exist_ok=True)
    return outpath


def main():
    env = Environment.get()
    env.enable_function_timers()

    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_benchmark (total)")

    mpiworld, procs, rank = get_world()

    if rank == 0:
        log.info("TOAST version = {}".format(env.version()))
        log.info("Using a maximum of {} threads per process".format(env.max_threads()))
    if mpiworld is None:
        log.info("Running serially with one process at {}".format(str(datetime.now())))
    else:
        if rank == 0:
            log.info(
                "Running with {} processes at {}".format(procs, str(datetime.now()))
            )

    cases = {
        "tiny": 5000000,  # O(1) GB RAM
        "xsmall": 50000000,  # O(10) GB RAM
        "small": 500000000,  # O(100) GB RAM
        "medium": 5000000000,  # O(1) TB RAM
        "large": 50000000000,  # O(10) TB RAM
        "xlarge": 500000000000,  # O(100) TB RAM
        "heroic": 5000000000000,  # O(1000) TB RAM
    }

    args, comm, n_nodes, n_detector, case, group_seconds, n_group = job_config(
        mpiworld, cases
    )

    # Note:  The number of "days" here will just be an approximation of the desired
    # data volume since we are doing a realistic schedule for a real observing site.

    n_days = int(2.0 * (group_seconds * n_group) / (24 * 3600))
    if n_days == 0:
        n_days = 1

    if rank == 0:
        log.info(
            "Using {} detectors for approximately {} days".format(n_detector, n_days)
        )

    # Create the schedule file and input maps on one process
    if rank == 0:
        create_schedules(args, group_seconds, n_days)
        create_input_maps(args)
    if mpiworld is not None:
        mpiworld.barrier()

    if args.dry_run is not None:
        if rank == 0:
            log.info("Exit from dry run")
        # We are done!
        sys.exit(0)

    gt.start("toast_benchmark (science work)")

    # Load and broadcast the schedule file

    schedules = pipeline_tools.load_schedule(args, comm)

    # Load the weather and append to schedules

    pipeline_tools.load_weather(args, comm, schedules)

    # Simulate the focalplane

    detweights = create_focalplanes(args, comm, schedules, n_detector)

    # Create the TOAST data object to match the schedule.  This will
    # include simulating the boresight pointing.

    data, telescope_data, total_samples = create_observations(args, comm, schedules)

    # handle = None
    # if comm.world_rank == 0:
    #     handle = open(os.path.join(args.outdir, "distdata.txt"), "w")
    # data.info(handle)
    # if comm.world_rank == 0:
    #     handle.close()
    # if comm.comm_world is not None:
    #     comm.comm_world.barrier()

    # Split the communicator for day and season mapmaking

    time_comms = pipeline_tools.get_time_communicators(args, comm, data)

    # Expand boresight quaternions into detector pointing weights and
    # pixel numbers

    pipeline_tools.expand_pointing(args, comm, data)

    # Optionally rewrite the noise PSD:s in each observation to include
    # elevation-dependence

    pipeline_tools.get_elevation_noise(args, comm, data)

    # Purge the pointing if we are NOT going to export the
    # data to a TIDAS volume
    if (args.tidas is None) and (args.spt3g is None):
        for ob in data.obs:
            tod = ob["tod"]
            tod.free_radec_quats()

    # Prepare auxiliary information for distributed map objects

    signalname = pipeline_tools.scan_sky_signal(args, comm, data, "signal")

    # Set up objects to take copies of the TOD at appropriate times

    totalname, totalname_freq = setup_sigcopy(args)

    # Loop over Monte Carlos

    firstmc = args.MC_start
    nsimu = args.MC_count

    freqs = [float(freq) for freq in args.freq.split(",")]
    nfreq = len(freqs)

    for mc in range(firstmc, firstmc + nsimu):

        pipeline_tools.simulate_atmosphere(args, comm, data, mc, totalname)

        # Loop over frequencies with identical focal planes and identical
        # atmospheric noise.

        for ifreq, freq in enumerate(freqs):

            if comm.world_rank == 0:
                log.info(
                    "Processing frequency {}GHz {} / {}, MC = {}".format(
                        freq, ifreq + 1, nfreq, mc
                    )
                )

            # Make a copy of the atmosphere so we can scramble the gains and apply
            # frequency-dependent scaling.
            pipeline_tools.copy_signal(args, comm, data, totalname, totalname_freq)

            pipeline_tools.scale_atmosphere_by_frequency(
                args, comm, data, freq=freq, mc=mc, cache_name=totalname_freq
            )

            pipeline_tools.update_atmospheric_noise_weights(args, comm, data, freq, mc)

            # Add previously simulated sky signal to the atmospheric noise.

            pipeline_tools.add_signal(
                args, comm, data, totalname_freq, signalname, purge=(nsimu == 1)
            )

            mcoffset = ifreq * 1000000

            pipeline_tools.simulate_noise(
                args, comm, data, mc + mcoffset, totalname_freq
            )

            pipeline_tools.scramble_gains(
                args, comm, data, mc + mcoffset, totalname_freq
            )

            outpath = setup_output(args, comm, mc + mcoffset, freq)

            # Bin and destripe maps

            pipeline_tools.apply_mapmaker(
                args,
                comm,
                data,
                outpath,
                totalname_freq,
                time_comms=time_comms,
                telescope_data=telescope_data,
                first_call=(mc == firstmc),
            )

            if args.apply_polyfilter or args.apply_groundfilter:

                # Filter signal

                pipeline_tools.apply_polyfilter(args, comm, data, totalname_freq)

                pipeline_tools.apply_groundfilter(args, comm, data, totalname_freq)

                # Bin filtered maps

                pipeline_tools.apply_mapmaker(
                    args,
                    comm,
                    data,
                    outpath,
                    totalname_freq,
                    time_comms=time_comms,
                    telescope_data=telescope_data,
                    first_call=False,
                    extra_prefix="filtered",
                    bin_only=True,
                )

    gt.stop_all()
    if mpiworld is not None:
        mpiworld.barrier()

    runtime = gt.seconds("toast_benchmark (science work)")
    kilo_samples = 1.0e-3 * total_samples
    factor = 1.10
    metric = (kilo_samples) ** factor / (runtime * n_nodes)
    if rank == 0:
        msg = "Science Metric: ({:0.3e})**({:0.3f}) / ({:0.1f} * {}) = {:0.2f}".format(
            kilo_samples, factor, runtime, n_nodes, metric
        )
        log.info("")
        log.info(msg)
        log.info("")
        with open(os.path.join(args.outdir, "log"), "a") as f:
            f.write(msg)
            f.write("\n\n")

    timer = Timer()
    timer.start()
    alltimers = gather_timers(comm=mpiworld)
    if comm.world_rank == 0:
        out = os.path.join(args.outdir, "timing")
        dump_timing(alltimers, out)
        with open(os.path.join(args.outdir, "log"), "a") as f:
            f.write("Copy of Global Timers:\n")
            with open("{}.csv".format(out), "r") as t:
                f.write(t.read())
        timer.stop()
        timer.report("Gather and dump timing info")
    return


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # We have an unhandled exception on at least one process.  Print a stack
        # trace for this process and then abort so that all processes terminate.
        mpiworld, procs, rank = get_world()
        if procs == 1:
            raise
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = ["Proc {}: {}".format(rank, x) for x in lines]
        print("".join(lines), flush=True)
        if mpiworld is not None:
            mpiworld.Abort(6)
