#!/usr/bin/env python3

# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This script finds observations listed in a TOAST schedule
that overlap with a provided target area.
"""

import argparse
import os
import sys
import traceback

import astropy.units as u
import ephem
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from toast.coordinates import to_DJD, to_UTC
from toast.mpi import MPI, Comm, get_world
from toast.pixels_io_healpix import read_healpix, write_healpix
from toast.timing import Timer
from toast.utils import Environment, Logger

import toast


# tqdm provides a progress bar but it is not critical
def no_tqdm(x):
    return x


try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = no_tqdm


def get_obs(args, schedule):
    """Construct lists of observation start/stop times and observation names"""

    tstart_schedule = schedule.scans[0].start.timestamp()
    tstop_schedule = schedule.scans[-1].stop.timestamp()
    tschedule = tstop_schedule - tstart_schedule

    obs_times = []
    obs_names = []
    for scan in schedule.scans:
        obs_times.append((scan.start.timestamp(), scan.stop.timestamp()))
        # Construct the observation name just like sim_ground.py
        sname = f"{scan.name}-{scan.scan_indx}-{scan.subscan_indx}"
        obs_names.append(sname)

    return obs_times, obs_names


def get_azel(args, scan):
    """Get a vector of azimuths and the elevation that covers the scan"""

    azstep = np.radians(args.azstep)
    azmin = scan.az_min.to_value(u.rad)
    azmax = scan.az_max.to_value(u.rad)
    el = scan.el.to_value(u.rad)
    if args.modulate:
        # Get az vector for the modulated scan strategy
        az = azmin
        azvec = []
        while az < azmax:
            azvec.append(az)
            dazdt = azstep / np.abs(np.sin(az))
            az += dazdt
    else:
        # Get az for an unmodulated scan
        azvec = np.arange(azmin, azmax, azstep)

    return azvec, el


def get_obs_hits(args, schedule, obs_times, iobs, iscan):
    """Make a hitmap for the given observation"""

    radius = np.radians(args.fov) / 2
    tstart, tstop = obs_times[iobs]
    npix = hp.nside2npix(args.nside)
    hits = np.zeros(npix)

    observer = ephem.Observer()
    observer.lon = schedule.site_lon.to_value(u.radian)
    observer.lat = schedule.site_lat.to_value(u.radian)
    observer.elevation = schedule.site_alt.to_value(u.meter)
    observer.epoch = ephem.J2000
    observer.compute_pressure()

    # Find all scans that overlap with this observation
    for t in np.arange(tstart, tstop, args.timestep):
        # Advance `iscan` until scan stop is after `t`
        while schedule.scans[iscan].stop.timestamp() < t:
            # This observation is before our reference time
            iscan += 1
        scan = schedule.scans[iscan]
        if scan.start.timestamp() < t:
            # `t` is between scan start and stop. Project hits according
            # to the azimuth range at our reference time
            observer.date = to_DJD(t)
            azvec, el = get_azel(args, scan)
            for az in azvec:
                ra, dec = observer.radec_of(az, el)
                vec = hp.dir2vec(np.pi / 2 - dec, ra)
                pix = hp.query_disc(args.nside, vec, radius=radius)
                hits[pix] += 1

    return hits, iscan


def get_hits(args, schedule, obs_times, comm, rank):
    """Build or load the hitmaps for each observation"""

    log = Logger.get()

    nobs = len(obs_times)
    npix = hp.nside2npix(args.nside)

    if args.cache is not None and os.path.isfile(args.cache):
        # Just load the cached hits
        hits = np.load(args.cache)
        log.info_rank(f"Loaded {args.cache}", comm=comm)
        nobs_hits, npix_hits = hits.shape
        if nobs_hits != nobs or npix_hits != npix:
            msg = f"{args.cache} is incompatible with arguments"
            raise RuntimeError(msg)
        return hits

    log.info_rank("Computing hitmaps", comm=comm)
    if rank == 0:
        counter = tqdm
    else:
        counter = no_tqdm
    hits = np.zeros([nobs, npix])
    iscan = 0  # Track the current observation
    for iobs in counter(range(nobs)):
        if comm is not None and iobs % comm.size != rank:
            continue
        hits[iobs], iscan = get_obs_hits(
            args, schedule, obs_times, iobs, iscan
        )
    if comm is not None:
        hits = comm.allreduce(hits)

    if rank == 0 and args.cache is not None:
        np.save(args.cache, hits)
        log.info_rank(f"Wrote {args.cache}", comm=comm)

    return hits


def get_mask(args):
    """Make a mask of the target pixels"""

    npix = hp.nside2npix(args.nside)
    mask = np.ones(npix, dtype=bool)
    pix = np.arange(npix)
    ra, dec = hp.pix2ang(args.nside, pix, lonlat=True)
    # Make sure we compare the right branch of RA
    ra_min = args.ra_min_deg
    ra_max = args.ra_max_deg
    while ra_min < 0:
        ra_min += 360
        ra_max += 360
    ra[ra < ra_min] += 360
    mask[ra > ra_max] = False
    # Dec is easier
    mask[dec < args.dec_min_deg] = False
    mask[dec > args.dec_max_deg] = False

    return mask

    
def parse_arguments():
    """Parse the command line arguments"""

    parser = argparse.ArgumentParser(
        description="Filter a TOAST schedule for observations that overlap with a specified target area."
    )

    parser.add_argument(
        "schedule",
        help="TOAST observing schedule",
    )

    parser.add_argument(
        "--out",
        default="overlapping_schedule.txt",
        help="Name of the output schedule",
    )

    parser.add_argument(
        "--fov",
        default=5,
        type=float,
        help="Field of view in degrees",
    )

    parser.add_argument(
        "--nside",
        default=128,
        type=int,
        help="Hitmap healpix resolution",
    )

    parser.add_argument(
        "--timestep",
        default=600,
        help="Time step in seconds (default is 10 minutes). "
        "Each observation is sampled once every time step.",
    )

    parser.add_argument(
        "--azstep",
        default=1,
        help="Width of azimuth step in degrees",
    )

    parser.add_argument(
        "--modulate",
        required=False,
        default=False,
        action="store_true",
        help="Simulate the azimuth-modulated scan strategy",
    )

    parser.add_argument(
        "--cache",
        required=False,
        help="Optional file for saving hits (numpy save file)",
    )

    parser.add_argument(
        "--ra-min-deg",
        default=0,
        type=float,
        help="Minimum RA of the target field",
    )

    parser.add_argument(
        "--ra-max-deg",
        default=360,
        type=float,
        help="Maximum RA of the target field",
    )

    parser.add_argument(
        "--dec-min-deg",
        default=-90,
        type=float,
        help="Minimum Dec of the target field",
    )

    parser.add_argument(
        "--dec-max-deg",
        default=90,
        type=float,
        help="Maximum Dec of the target field",
    )

    args = parser.parse_args()
    return args


def main():
    env = Environment.get()
    log = Logger.get()
    comm, ntask, rank = get_world()
    timer0 = Timer()
    timer1 = Timer()
    timer0.start()
    timer1.start()

    args = parse_arguments()

    # Load the observing schedule

    schedule = toast.schedule.GroundSchedule()
    schedule.read(args.schedule)

    log.info_rank(f"Loaded {args.schedule} in", timer=timer1, comm=comm)

    # Compute hitmaps for each observation

    obs_times, obs_names = get_obs(args, schedule)

    all_hits = get_hits(args, schedule, obs_times, comm, rank)
    log.info_rank(f"Made hits in", timer=timer1, comm=comm)

    if rank == 0:

        # Find overlaps

        mask = get_mask(args)
        input_schedule = open(args.schedule, "r").readlines()

        with open(args.out, "w") as f:
            # Parse header lines
            header_not_read = True
            while input_schedule[0].startswith("#") or header_not_read:
                if not input_schedule[0].startswith("#"):
                    header_not_read = False
                f.write(input_schedule[0])
                del input_schedule[0]
            # Remove any potential comment lines later in the schedule
            iline = 0
            while iline < len(input_schedule):
                if input_schedule[iline].startswith("#"):
                    del input_schedule[iline]
                else:
                    iline += 1
            # Copy over all overlapping entries
            for iobs, hits in enumerate(all_hits):
                if np.any(hits[mask] != 0):
                    f.write(input_schedule[iobs])

        print(f"Wrote {args.out}")

    log.info_rank(f"Compared hitmaps in", timer=timer1, comm=comm)

    if comm is not None:
        comm.Barrier()

    log.info_rank(f"All done in", timer=timer0, comm=comm)

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
