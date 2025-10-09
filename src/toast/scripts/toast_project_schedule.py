#!/usr/bin/env python3

# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This script projects observations listed in a TOAST schedule
onto a hitmap.
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


def get_periods(args, schedule):
    """Construct lists of period start/stop times and period names"""

    tstart_schedule = schedule.scans[0].start.timestamp()
    tstop_schedule = schedule.scans[-1].stop.timestamp()
    tschedule = tstop_schedule - tstart_schedule

    period_times = []
    period_names = []
    if args.by_obs:
        # Each period covers exactly one observation
        for scan in schedule.scans:
            period_times.append((scan.start.timestamp(), scan.stop.timestamp()))
            # Construct the observation name just like sim_ground.py
            sname = f"{scan.name}-{scan.scan_indx}-{scan.subscan_indx}"
            period_names.append(sname)
    else:
        tperiod = args.period
        nperiod = int(np.ceil(tschedule / tperiod))
        for iperiod in range(nperiod):
            tstart_period = tstart_schedule + iperiod * tperiod
            tstop_period = min(tstart_period + tperiod, tstop_schedule)
            period_times.append((tstart_period, tstop_period))
            period_names.append(f"Period {iperiod:04}")

    return period_times, period_names


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


def get_period_hits(args, schedule, period_times, iperiod, iscan):
    """Make a hitmap for the given period"""

    radius = np.radians(args.fov) / 2
    tstart, tstop = period_times[iperiod]
    npix = hp.nside2npix(args.nside)
    hits = np.zeros(npix)

    observer = ephem.Observer()
    observer.lon = schedule.site_lon.to_value(u.radian)
    observer.lat = schedule.site_lat.to_value(u.radian)
    observer.elevation = schedule.site_alt.to_value(u.meter)
    observer.epoch = ephem.J2000
    observer.compute_pressure()

    # Find all scans that overlap with this period
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


def get_sso_period_hits(args, schedule, period_times, iperiod, iscan, sso, radius):
    """Make an SSO hitmap for the given period"""

    radius = np.radians(radius)
    tstart, tstop = period_times[iperiod]
    npix = hp.nside2npix(args.nside)
    hits = np.zeros(npix)

    observer = ephem.Observer()
    observer.lon = schedule.site_lon.to_value(u.radian)
    observer.lat = schedule.site_lat.to_value(u.radian)
    observer.elevation = schedule.site_alt.to_value(u.meter)
    observer.epoch = ephem.J2000
    observer.compute_pressure()

    sso = getattr(ephem, sso)()

    # Find all scans that overlap with this period
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
            sso.compute(observer)
            lon = np.degrees(sso.a_ra)
            lat = np.degrees(sso.a_dec)
            vec = hp.dir2vec(lon, lat, lonlat=True)
            pix = hp.query_disc(args.nside, vec, radius=radius)
            hits[pix] += 1

    return hits, iscan


def get_hits(args, schedule, period_times, comm, rank):
    """Build or load the hitmaps for each period"""

    log = Logger.get()

    nperiod = len(period_times)
    npix = hp.nside2npix(args.nside)

    if args.cache is not None and os.path.isfile(args.cache):
        # Just load the cached hits
        hits = np.load(args.cache)
        log.info_rank(f"Loaded {args.cache}", comm=comm)
        nperiod_hits, npix_hits = hits.shape
        if nperiod_hits != nperiod or npix_hits != npix:
            msg = f"{args.cache} is incompatible with arguments"
            raise RuntimeError(msg)
        return hits

    log.info_rank("Computing hitmaps", comm=comm)
    if rank == 0:
        counter = tqdm
    else:
        counter = no_tqdm
    hits = np.zeros([nperiod, npix])
    iscan = 0  # Track the current observation
    for iperiod in counter(range(nperiod)):
        if comm is not None and iperiod % comm.size != rank:
            continue
        hits[iperiod], iscan = get_period_hits(
            args, schedule, period_times, iperiod, iscan
        )
    if comm is not None:
        hits = comm.allreduce(hits)

    if rank == 0 and args.cache is not None:
        np.save(args.cache, hits)
        log.info_rank(f"Wrote {args.cache}", comm=comm)

    return hits


def get_sso(args, schedule, period_times, comm, rank, sso, radius):
    """Build or load SSO hitmaps for each period"""

    log = Logger.get()

    nperiod = len(period_times)
    npix = hp.nside2npix(args.nside)

    cachefile = args.cache
    if cachefile is not None:
        cachefile = cachefile.replace(".npy", f".{sso}.npy")
    if cachefile is not None and os.path.isfile(cachefile):
        # Just load the cached hits
        hits = np.load(cachefile)
        log.info_rank(f"Loaded {cachefile}", comm=comm)
        nperiod_hits, npix_hits = hits.shape
        if nperiod_hits != nperiod or npix_hits != npix:
            msg = f"{cachefile} is incompatible with arguments"
            raise RuntimeError(msg)
        return hits

    log.info_rank(f"Computing {sso} hitmaps", comm=comm)
    if rank == 0:
        counter = tqdm
    else:
        counter = no_tqdm
    hits = np.zeros([nperiod, npix])
    iscan = 0  # Track the current observation
    for iperiod in counter(range(nperiod)):
        if comm is not None and iperiod % comm.size != rank:
            continue
        hits[iperiod], iscan = get_sso_period_hits(
            args, schedule, period_times, iperiod, iscan, sso, radius
        )
    if comm is not None:
        hits = comm.allreduce(hits)

    if rank == 0 and cachefile is not None:
        np.save(cachefile, hits)
        log.info_rank(f"Wrote {cachefile}", comm=comm)

    return hits


def load_background(args):
    if args.bg is None:
        return None

    if args.bg_pol:
        bg = hp.read_map(args.bg, [0, 1, 2])
    else:
        bg = hp.read_map(args.bg)

    if args.bg_fwhm is not None:
        bg = hp.smoothing(bg, fwhm=np.radians(args.bg_fwhm), lmax=args.bg_lmax)

    if args.bg_pol:
        bg = np.sqrt(bg[1] ** 2 + bg[2] ** 2)

    # truncate the color scale
    limit = np.percentile(bg, args.bg_percentile)
    bg[bg > limit] = limit

    return bg


def plot_hits(args, all_hits, sso_hits, period_times, period_names, comm, rank):
    """
    Plot daily and total hitmaps for the provided schedule.

    If SSO (Solar System Object) hitmaps are provided via `sso_hits`, overlays are
    generated and normalized for visualization on top of the main hitmaps.
    This helps to highlight the coverage and visibility of SSOs during each period.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    all_hits : np.ndarray
        Array of hit counts per pixel for each period.
    sso_hits : np.ndarray or None
        Array of SSO hit counts per pixel for each period, or None if not used.
    period_times : list
        List of times for each period.
    period_names : list
        List of names for each period.
    comm : MPI.Comm or None
        MPI communicator.
    rank : int
        MPI rank of the current process.
    """
    log = Logger.get()

    bg = load_background(args)

    log.info_rank("Plotting hitmaps", comm=comm)
    if rank == 0:
        print("Plotting hitmaps")
        counter = tqdm
    else:
        counter = no_tqdm
    nperiod, npix = all_hits.shape
    for iperiod in counter(range(nperiod + 1)):
        if comm is not None and iperiod % comm.size != rank:
            continue
        if iperiod < nperiod:
            if args.no_step_plots:
                continue
            fname_plot = f"hits_{iperiod:04}.png"
            hits = all_hits[iperiod].copy()
            tstart, tstop = period_times[iperiod]
            name = period_names[iperiod]
        else:
            fname_plot = f"hits_tot.png"
            hits = np.sum(all_hits, 0)
            tstart = period_times[0][0]
            tstop = period_times[-1][1]
            name = "Full"
        title = f"{name} : {to_UTC(tstart)} - {to_UTC(tstop)}"
        mask = hits > 0
        hits[hits == 0] = hp.UNSEEN
        rot = [args.rot_ra, args.rot_dec]
        if bg is not None:
            hp.mollview(
                bg,
                cmap="inferno",
                rot=rot,
                coord=args.bg_coord,
                cbar=False,
                alpha=np.ones(bg.size) * args.bg_alpha,
            )
            reuse = True
            alpha = mask * 0.75
        else:
            reuse = False
            alpha = None
        hp.mollview(
            hits,
            cmap="magma",
            title=title,
            xsize=1600,
            unit="Hits",
            rot=rot,
            reuse_axes=reuse,
            alpha=alpha,
        )
        # Plot SSOs
        npix = hp.nside2npix(args.nside)
        lon, lat = hp.pix2ang(args.nside, np.arange(npix), lonlat=True)
        for sso, hits in sso_hits.items():
            if iperiod < nperiod:
                disc = hits[iperiod]
            else:
                disc = np.sum(hits, 0)
            if np.all(disc == 0):
                continue
            disc /= np.amax(disc)
            hp.mollview(
                disc,
                title=title,
                cmap="jet",
                rot=rot,
                reuse_axes=True,
                cbar=False,
                alpha=(disc != 0) * 0.5,
            )
            # Label the SSO
            hit_lat = lat.astype(float, copy=True)
            hit_lat[disc == 0] = hp.UNSEEN
            i = np.argmax(hit_lat)
            top_lon = lon[i]
            top_lat = lat[i]
            hp.projtext(top_lon, top_lat, sso, lonlat=True)

        plt.savefig(fname_plot)
        plt.close()
    return


def parse_arguments():
    """Parse the command line arguments"""

    parser = argparse.ArgumentParser(description="Project schedule to a hitmap")

    parser.add_argument(
        "schedule",
        help="TOAST observing schedule",
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=5,
        help="Field of view in degrees",
    )

    parser.add_argument(
        "--nside",
        type=int,
        default=128,
        help="Hitmap healpix resolution",
    )

    parser.add_argument(
        "--bg",
        required=False,
        help="Background map to plot",
    )

    parser.add_argument(
        "--bg-fwhm",
        type=float,
        required=False,
        help="Background smoothing scale in degrees",
    )

    parser.add_argument(
        "--bg-lmax",
        type=int,
        default=512,
        help="Background smoothing lmax",
    )

    parser.add_argument(
        "--bg-coord",
        required=False,
        help="Background coordinates passed to Healpy.",
    )

    parser.add_argument(
        "--bg-alpha",
        type=float,
        default=0.5,
        help="Background alpha passed to Healpy.",
    )

    parser.add_argument(
        "--bg-percentile",
        type=float,
        default=75,
        help="Saturate background colorscale at this percentile",
    )

    parser.add_argument(
        "--rot-ra",
        type=float,
        default=0,
        help="Rotate images in RA",
    )

    parser.add_argument(
        "--rot-dec",
        type=float,
        default=0,
        help="Rotate images in Declination",
    )

    parser.add_argument(
        "--bg-pol",
        required=False,
        default=False,
        action="store_true",
        help="Plot background polarization amplitude rather than intensity",
    )

    parser.add_argument(
        "--timestep",
        type=float,
        default=600,
        help="Time step in seconds (default is 10 minutes). "
        "Each period is sampled once every time step.",
    )

    parser.add_argument(
        "--by-obs",
        required=False,
        default=False,
        action="store_true",
        help="Rather than plotting time periods, plot each observation.",
    )

    parser.add_argument(
        "--period",
        type=float,
        default=86400,
        help="Time period in seconds (default is 1 day)",
    )

    parser.add_argument(
        "--azstep",
        type=float,
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
        help="Optional file for saving hits (numpy save file).  "
        "Must have a '.npy' suffix",
    )

    parser.add_argument(
        "--sso",
        required=False,
        help="Comma-separated list of solar system objects (SSOs) to plot. "
        "Use --sso-radius-deg to set the radius of the field",
    )

    parser.add_argument(
        "--sso-radius-deg",
        required=False,
        help="Comma-separated list of SSO radii.  If no commas are found, the "
        "same radius is applied to all SSOs.  See --sso.",
    )

    parser.add_argument(
        "--no-step-plots",
        required=False,
        default=False,
        action="store_true",
        help="Skip plotting step hit maps and only provide the total",
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

    if args.cache is not None and not args.cache.endswith(".npy"):
        msg = f"Cache file does not end with .npy: {args.cache}"
        raise RuntimeError(msg)

    if args.sso is not None:
        if args.sso_radius_deg is None:
            raise RuntimeError("--sso-radius-deg is required when using --sso")
        nsso = len(args.sso.split(","))
        radii = args.sso_radius_deg.split(",")
        if len(radii) not in (1, nsso):
            msg = "Number of radii must be 1 or match number of SSOs but "
            msg += f"{len(radii)} != {nsso}"
            raise RuntimeError(msg)

    # Load the observing schedule

    schedule = toast.schedule.GroundSchedule()
    schedule.read(args.schedule)

    log.info_rank(f"Loaded {args.schedule} in", timer=timer1, comm=comm)

    # Compute hitmaps for each period

    period_times, period_names = get_periods(args, schedule)

    all_hits = get_hits(args, schedule, period_times, comm, rank)
    log.info_rank(f"Made hits in", timer=timer1, comm=comm)

    sso_hits = {}
    if args.sso is not None:
        for isso, sso in enumerate(args.sso.split(",")):
            radii = args.sso_radius_deg.split(",")
            if len(radii) == 1:
                radius = float(radii[0])
            else:
                radius = float(radii[isso])
            print(f"Projecting {sso} at radius {radius} deg")
            sso_hits[sso] = get_sso(
                args, schedule, period_times, comm, rank, sso, radius
            )
            log.info_rank(f"Made {sso} hits in", timer=timer1, comm=comm)

    # Plot

    plot_hits(args, all_hits, sso_hits, period_times, period_names, comm, rank)
    log.info_rank(f"Made plots in", timer=timer1, comm=comm)

    if comm is not None:
        comm.Barrier()

    log.info_rank(f"All done in", timer=timer0, comm=comm)

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
