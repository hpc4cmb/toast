#!/usr/bin/env python3
"""
Use PySM to pre-compute a detector signal map that we can then scan from
in parallel.
"""

import argparse

import numpy as np

import healpy as hp

import pysm
import pysm.units as u

parser = argparse.ArgumentParser(description="Run PySM to build a sky map.")

parser.add_argument(
    "--output", required=False, default=None, help="The output map FITS file.",
)

parser.add_argument(
    "--nside", required=True, default=None, type=int, help="The map NSIDE resolution",
)

parser.add_argument(
    "--beam_arcmin",
    required=True,
    default=None,
    type=float,
    help="The beam FWHM in arcmin",
)

parser.add_argument(
    "--bandcenter_ghz",
    required=True,
    default=None,
    type=float,
    help="The band center in GHz",
)

parser.add_argument(
    "--bandwidth_ghz",
    required=True,
    default=None,
    type=float,
    help="The band width in GHz",
)

parser.add_argument(
    "--coord",
    required=True,
    default=None,
    type=str,
    help="The coordinate system (G, E, C)",
)

args = parser.parse_args()

outfile = args.output
if outfile is None:
    outfile = "input_sky_band-{:.1f}-{:.1f}_beam-{:.1f}_n{:02d}-{}.fits".format(
        args.bandcenter_ghz,
        args.bandwidth_ghz,
        args.beam_arcmin,
        args.nside,
        args.coord,
    )

print("Working on {}".format(outfile))

print("  Construct Sky...", flush=True)
sky = pysm.Sky(nside=args.nside, preset_strings=["d1", "s1", "f1", "a1"])

freqs = (
    np.array(
        [
            args.bandcenter_ghz - 0.5 * args.bandwidth_ghz,
            args.bandcenter_ghz,
            args.bandcenter_ghz + 0.5 * args.bandwidth_ghz,
        ]
    )
    * u.GHz
)

print("  Get Emission...", flush=True)
map_data = sky.get_emission(freqs)

if args.coord == "G":
    print("  Smoothing...", flush=True)
    smoothed = pysm.apply_smoothing_and_coord_transform(
        map_data, fwhm=args.beam_arcmin * u.arcmin
    )
else:
    to_coord = "G{}".format(args.coord)
    print("  Smoothing and rotating...", flush=True)
    smoothed = pysm.apply_smoothing_and_coord_transform(
        map_data, fwhm=args.beam_arcmin * u.arcmin, rot=hp.Rotator(coord=to_coord)
    )

print("  Converting to NEST")
nested = hp.ud_grade(smoothed, args.nside, order_in="RING", order_out="NEST")

print("  Writing output...", flush=True)
hp.write_map(outfile, nested, nest=True)
