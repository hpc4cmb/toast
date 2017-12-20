#!/usr/bin/env python3

# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# This generates a focalplane pickle file compatible with the
# example pipelines.

import pickle
import argparse

import numpy as np
from scipy.constants import degree

import toast.tod as tt
import toast.timing as timing

parser = argparse.ArgumentParser(
    description="Simulate fake hexagonal focalplane.",
    fromfile_prefix_chars='@')

parser.add_argument( "--minpix", required=False, type=int, default=100,
                     help="minimum number of pixels to use" )

parser.add_argument( "--out", required=False, default="fp_fake",
                     help="Root name of output pickle file" )

parser.add_argument( "--fwhm", required=False, type=float, default=5.0,
                     help="beam FWHM in arcmin" )

parser.add_argument( "--fov", required=False, type=float, default=5.0,
                     help="Field of View in degrees" )

parser.add_argument( "--psd_fknee", required=False, type=float, default=0.05,
                     help="Detector noise model f_knee" )

parser.add_argument( "--psd_NET", required=False, type=float, default=60.0e-6,
                     help="Detector noise model NET" )

parser.add_argument( "--psd_alpha", required=False, type=float, default=1.0,
                     help="Detector noise model slope" )

parser.add_argument( "--psd_fmin", required=False, type=float, default=1.0e-5,
                     help="Detector noise model f_min" )

parser.add_argument( "--bandcenter_ghz", required=False, type=float, default=25,
                     help="Band center frequency [GHz]" )

parser.add_argument( "--bandwidth_ghz", required=False, type=float, default=6,
                     help="Bandwidth [GHz]" )

args = timing.add_arguments_and_parse(parser, timing.FILE(noquotes=True))

autotimer = timing.auto_timer(timing.FILE())

# Make one big hexagon layout at the center of the focalplane.
# Compute the number of pixels that is at least the number requested.

test = args.minpix - 1
nrings = 0
while (test - 6 * nrings) > 0:
    test -= 6 * nrings
    nrings += 1

npix = 1
for r in range(1, nrings+1):
    npix += 6 * r

print("using {} pixels ({} detectors)".format(npix, npix*2))

fwhm = args.fwhm / 60.0
width = 100.0
# Translate the field-of-view into distance between flag sides
angwidth = args.fov * np.cos(30 * degree)

Apol = tt.hex_pol_angles_qu(npix, offset=0.0)
Bpol = tt.hex_pol_angles_qu(npix, offset=90.0)

Adets = tt.hex_layout(npix, width, angwidth, fwhm, "fake_", "A", Apol)
Bdets = tt.hex_layout(npix, width, angwidth, fwhm, "fake_", "B", Bpol)

dets = Adets.copy()
dets.update(Bdets)

for indx, d in enumerate(sorted(dets.keys())):
    dets[d]["fknee"] = args.psd_fknee
    dets[d]["fmin"] = args.psd_fmin
    dets[d]["alpha"] = args.psd_alpha
    dets[d]["NET"] = args.psd_NET
    dets[d]["fwhm_deg"] = fwhm  # in degrees
    dets[d]["bandcenter_ghz"] = args.bandcenter_ghz
    dets[d]["bandwidth_ghz"] = args.bandwidth_ghz
    dets[d]["index"] = indx

outfile = "{}_{}".format(args.out, npix)
tt.plot_focalplane(dets, args.fov, args.fov, "{}.png".format(outfile))

with open("{}.pkl".format(outfile), "wb") as p:
    pickle.dump(dets, p)

tman = timing.timing_manager()
tman.report()

tman.clear()
