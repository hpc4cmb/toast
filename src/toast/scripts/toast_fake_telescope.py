#!/usr/bin/env python3

# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""Generate a synthetic telescope file compatible with SimGround and SimSatellite."""

import os
import argparse
from datetime import datetime, timezone

import numpy as np
from astropy import units as u

import toast
from toast.instrument import GroundSite, SpaceSite, Telescope
from toast.instrument_sim import fake_hexagon_focalplane, plot_focalplane
from toast.io.observation_hdf_save import save_instrument_file
from toast.mpi import get_world
from toast.utils import Logger
from toast.weather import SimWeather


def main(opts=None):
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Create a synthetic ground or space telescope."
    )

    parser.add_argument(
        "--telescope_name",
        required=False,
        default="telescope",
        help="Name of the telescope",
    )

    parser.add_argument(
        "--time",
        required=False,
        default=None,
        help="The ISO date / time string (e.g. 2026-10-27T10:30:00-04:00)",
    )

    parser.add_argument(
        "--ground_site_name",
        required=False,
        default=None,
        help="For ground-based telescopes, the name of the site",
    )

    loc_help = "For ground-based telescopes, the 'lon,lat,alt' in degrees and meters."
    loc_help += " Other supported names are: 'toco', 'chajnantor', 'LMT', and 'pole'"
    parser.add_argument(
        "--ground_site_loc",
        required=False,
        default=None,
        help=loc_help,
    )

    weather_help = "For ground-based telescopes, the name / path of the weather file."
    weather_help += " If a known site name is used for `ground_site_loc`, then the"
    weather_help += " correct weather name will be used."
    parser.add_argument(
        "--ground_weather",
        required=False,
        default=None,
        help=weather_help,
    )

    parser.add_argument(
        "--min_pix",
        required=False,
        type=int,
        default=7,
        help="Choose a hexagonal packing with at least this many pixels",
    )

    parser.add_argument(
        "--out",
        required=False,
        default="telescope.h5",
        help="Name of output HDF5 file",
    )

    parser.add_argument(
        "--fwhm_arcmin",
        required=False,
        type=float,
        default=10.0,
        help="beam FWHM in arcmin",
    )

    parser.add_argument(
        "--fwhm_sigma",
        required=False,
        type=float,
        default=0,
        help="Relative beam FWHM distribution width",
    )

    parser.add_argument(
        "--fov_deg",
        required=False,
        type=float,
        default=5.0,
        help="Field of View in degrees",
    )

    parser.add_argument(
        "--sample_rate",
        required=False,
        type=float,
        default=50.0,
        help="Detector sample rate in Hz",
    )

    parser.add_argument(
        "--pol_leakage",
        required=False,
        type=float,
        default=0.0,
        help="Detector polarization leakage (0 = perfect polarizer)",
    )

    parser.add_argument(
        "--psd_fknee",
        required=False,
        type=float,
        default=0.05,
        help="Detector noise model f_knee in Hz",
    )

    parser.add_argument(
        "--psd_net",
        required=False,
        type=float,
        default=50.0e-6,
        help="Detector noise model NET in K*sqrt(sec)",
    )

    parser.add_argument(
        "--psd_alpha",
        required=False,
        type=float,
        default=1.0,
        help="Detector noise model slope",
    )

    parser.add_argument(
        "--psd_fmin",
        required=False,
        type=float,
        default=1.0e-5,
        help="Detector noise model f_min in Hz",
    )

    parser.add_argument(
        "--bandcenter_ghz",
        required=False,
        type=float,
        default=150.0,
        help="Band center frequency in GHz",
    )

    parser.add_argument(
        "--bandcenter_sigma",
        required=False,
        type=float,
        default=0,
        help="Band center distribution width in GHz",
    )

    parser.add_argument(
        "--bandwidth_ghz",
        required=False,
        type=float,
        default=20.0,
        help="Bandwidth in GHz",
    )

    parser.add_argument(
        "--bandwidth_sigma",
        required=False,
        type=float,
        default=0,
        help="Bandwidth distribution width in GHz",
    )

    parser.add_argument(
        "--random_seed",
        required=False,
        type=int,
        default=123456,
        help="Random number generator seed for randomized detector parameters",
    )

    args = parser.parse_args(args=opts)

    log = Logger.get()

    # Get the observing time for this telescope
    if args.time is None:
        obs_time = datetime.now(timezone.utc)
    else:
        obs_time = datetime.fromisoformat(args.time)

    # Construct the Site

    def _site_name_and_weather(loc_name):
        if args.ground_site_name is None:
            site_name = loc_name
        else:
            site_name = args.ground_site_name
        if args.ground_weather is None:
            if loc_name == "toco" or loc_name == "chajnantor":
                weather = SimWeather(time=obs_time, name="atacama", median_weather=True)
            elif loc_name == "LMT":
                weather = SimWeather(time=obs_time, name="LMT", median_weather=True)
            elif loc_name == "pole":
                weather = SimWeather(
                    time=obs_time, name="south_pole", median_weather=True
                )
            else:
                msg = "Non-standard ground location and no weather specified."
                log.warning(msg)
                weather = None
        else:
            if os.path.isfile(args.ground_weather):
                weather = SimWeather(
                    time=obs_time, file=args.ground_weather, median_weather=True
                )
            else:
                weather = SimWeather(
                    time=obs_time, name=args.ground_weather, median_weather=True
                )
        return site_name, weather

    if args.ground_site_loc is None:
        # Space site.  Use the telescope name as the site name.
        site = SpaceSite(args.telescope_name)
    else:
        if args.ground_site_loc == "toco":
            lon = -67.78797 * u.degree
            lat = -22.96047 * u.degree
            alt = 5188.0 * u.meter
            loc_name = args.ground_site_loc
        elif args.ground_site_loc == "chajnantor":
            lon = -67.74116 * u.degree
            lat = -22.98579 * u.degree
            alt = 5620.0 * u.meter
            loc_name = args.ground_site_loc
        elif args.ground_site_loc == "LMT":
            lon = -97.31470 * u.degree
            lat = 18.98586 * u.degree
            alt = 4600.0 * u.meter
            loc_name = args.ground_site_loc
        elif args.ground_site_loc == "pole":
            lon = 45.0 * u.degree
            lat = -89.9894 * u.degree
            alt = 2800.0 * u.meter
            loc_name = args.ground_site_loc
        else:
            # Must be exact location
            lon, lat, alt = args.ground_site_loc.split(",")
            lon = lon * u.degree
            lat = lat * u.degree
            alt = alt * u.meter
            loc_name = "unknown"
        site_name, weather = _site_name_and_weather(loc_name)
        site = GroundSite(site_name, lat, lon, alt, weather=weather)

    npix = 1
    ring = 1
    while npix < args.min_pix:
        npix += 6 * ring
        ring += 1

    msg = "Using {} hexagon-packed pixels, which is >= requested number ({})".format(
        npix, args.min_pix
    )
    log.info(msg)

    fp = fake_hexagon_focalplane(
        n_pix=npix,
        width=args.fov_deg * u.degree,
        sample_rate=args.sample_rate * u.Hz,
        epsilon=args.pol_leakage,
        fwhm=args.fwhm_arcmin * u.arcmin,
        bandcenter=args.bandcenter_ghz * u.GHz,
        bandwidth=args.bandwidth_ghz * u.GHz,
        psd_net=args.psd_net * u.K * np.sqrt(1 * u.second),
        psd_fmin=args.psd_fmin * u.Hz,
        psd_alpha=args.psd_alpha,
        psd_fknee=args.psd_fknee * u.Hz,
        fwhm_sigma=args.fwhm_sigma * u.arcmin,
        bandcenter_sigma=args.bandcenter_sigma * u.GHz,
        bandwidth_sigma=args.bandwidth_sigma * u.GHz,
        random_seed=args.random_seed,
    )

    tele = Telescope(args.telescope_name, site=site, focalplane=fp)

    # Guard against being called with multiple processes
    mpiworld, procs, rank = get_world()
    if rank == 0:
        save_instrument_file(args.out, tele, None)
        plotfile = "{}.pdf".format(args.out)
        plot_focalplane(
            fp,
            args.fov_deg * u.degree,
            args.fov_deg * u.degree,
            plotfile,
            show_labels=True,
        )


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
