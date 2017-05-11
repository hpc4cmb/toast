#!/usr/bin/env python

# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

# This generates a focalplane pickle file compatible with the
# example pipelines.

import pickle
import argparse

import numpy as np

import toast.tod as tt

parser = argparse.ArgumentParser( description="Simulate fake hexagonal focalplane." )

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
    
args = parser.parse_args()

# Make one big hexagon layout at the center of the focalplane.
# Compute the number of pixels that is at least the number requested.

test = args.minpix - 1
nrings = 1
while (test - 6 * nrings) >= 0:
    test -= 6 * nrings
    nrings += 1

npix = 1
for r in range(1, nrings+1):
    npix += 6 * r

print("using {} pixels ({} detectors)".format(npix, npix*2))

fwhm = args.fwhm / 60.0
width = 100.0
angwidth = args.fov

Apol = tt.hex_pol_angles_qu(npix, offset=0.0)
Bpol = tt.hex_pol_angles_qu(npix, offset=90.0)

Adets = tt.hex_layout(npix, width, angwidth, fwhm, "fake_", "A", Apol)
Bdets = tt.hex_layout(npix, width, angwidth, fwhm, "fake_", "B", Bpol)

dets = Adets.copy()
dets.update(Bdets)

indx = 0
for d in sorted(dets.keys()):
    dets[d]["fknee"] = 0.05
    dets[d]["fmin"] = 1.0e-5
    dets[d]["alpha"] = 1.0
    dets[d]["NET"] = 60.0e-6
    dets[d]["index"] = indx
    indx += 1

outfile="{}_{}".format(args.out, npix)
tt.plot_focalplane(dets, 6.0, 6.0, "{}.png".format(outfile))

with open("{}.pkl".format(outfile), "wb") as p:
    pickle.dump(dets, p)

