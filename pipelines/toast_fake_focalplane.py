#!/usr/bin/env python

# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

# This generates a focalplane pickle file compatible with the
# example pipelines.

import pickle
import numpy as np

import toast.tod as tt

# Make one big hexagon layout at the center of the focalplane.
# Set the FWHM to be 5 arcmin, a 5 degree FOV, and 631 pixels.

npix = 631
fwhm = 5.0 / 60.0
width = 100.0
angwidth = 5.0

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

tt.plot_focalplane(dets, 6.0, 6.0, "fp_fake.png")

with open("fp_fake.pkl", "wb") as p:
    pickle.dump(dets, p)

