# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from ..mpi import MPI
from .mpi import MPITestCase

from ..tod import sim_focalplane as sfp


def generate_hex(npix, width, poltype, fwhm):
    if poltype == "qu":
        pol_a = sfp.hex_pol_angles_qu(npix)
        pol_b = sfp.hex_pol_angles_qu(npix, offset=90.0)
    elif poltype == "radial":
        pol_a = sfp.hex_pol_angles_radial(npix)
        pol_b = sfp.hex_pol_angles_radial(npix, offset=90.0)
    dets_a = sfp.hex_layout(npix, width, "", "A", pol_a)
    dets_b = sfp.hex_layout(npix, width, "", "B", pol_b)

    dets = dict()
    dets.update(dets_a)
    dets.update(dets_b)

    # Pol color different for A/B detectors
    detpolcolor = dict()
    detpolcolor.update({ x : "red" for x in dets_a.keys() })
    detpolcolor.update({ x : "blue" for x in dets_b.keys() })

    # set the label to just the detector name
    detlabels = { x : x for x in dets.keys() }

    # fwhm and face color the same
    detfwhm = { x : fwhm for x in dets.keys() }

    # cycle through some colors just for fun
    pclr = [
        (1.0, 0.0, 0.0, 0.1),
        (1.0, 0.5, 0.0, 0.1),
        (0.25, 0.5, 1.0, 0.1),
        (0.0, 0.75, 0.0, 0.1)
    ]
    detcolor = { y : pclr[(x // 2) % 4] for x, y in \
        enumerate(sorted(dets.keys())) }

    # split out quaternions for plotting
    detquats = { x : dets[x]["quat"] for x in dets.keys() }

    return dets, detquats, detfwhm, detcolor, detpolcolor, detlabels


def generate_rhombus(npix, width, fwhm, prefix, center):
    pol_a = sfp.rhomb_pol_angles_qu(npix)
    pol_b = sfp.rhomb_pol_angles_qu(npix, offset=90.0)
    dets_a = sfp.rhombus_layout(npix, width, prefix, "A", pol_a, center=center)
    dets_b = sfp.rhombus_layout(npix, width, prefix, "B", pol_b, center=center)

    dets = dict()
    dets.update(dets_a)
    dets.update(dets_b)

    # Pol color different for A/B detectors
    detpolcolor = dict()
    detpolcolor.update({ x : "red" for x in dets_a.keys() })
    detpolcolor.update({ x : "blue" for x in dets_b.keys() })

    # set the label to just the detector name
    detlabels = { x : x for x in dets.keys() }

    # fwhm and face color the same
    detfwhm = { x : fwhm for x in dets.keys() }

    # cycle through some colors just for fun
    pclr = [
        (1.0, 0.0, 0.0, 0.1),
        (1.0, 0.5, 0.0, 0.1),
        (0.25, 0.5, 1.0, 0.1),
        (0.0, 0.75, 0.0, 0.1)
    ]
    detcolor = { y : pclr[(x // 2) % 4] for x, y in \
        enumerate(sorted(dets.keys())) }

    # split out quaternions for plotting
    detquats = { x : dets[x]["quat"] for x in dets.keys() }

    return dets, detquats, detfwhm, detcolor, detpolcolor, detlabels


class SimFocalplaneTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)


    def test_cart_quat(self):
        xincr = np.linspace(-5.0, 5.0, num=10, endpoint=True)
        yincr = np.linspace(-5.0, 5.0, num=10, endpoint=True)
        offsets = list()
        for x in xincr:
            for y in yincr:
                ang = 3.6 * (x - xincr[0]) * (y - yincr[0])
                offsets.append([x, y, ang])
        quats = sfp.cartesian_to_quat(offsets)
        detquats = { "{}".format(x) : y for x, y in enumerate(quats) }
        fwhm = { x : 30.0 for x in detquats.keys() }
        outfile = os.path.join(self.outdir, "out_test_cart2quat.png")
        sfp.plot_focalplane(detquats, 12.0, 12.0, outfile, fwhm=fwhm)
        return


    def test_hex_nring(self):
        result = {
            1 : 1,
            7 : 2,
            19 : 3,
            37 : 4,
            61 : 5,
            91 : 6,
            127 : 7,
            169 : 8,
            217 : 9,
            271 : 10,
            331 : 11,
            397 : 12
        }
        for npix, check in result.items():
            test = sfp.hex_nring(npix)
            nt.assert_equal(test, check)
        return


    def test_vis_hex_small(self):
        dets, detquats, detfwhm, detcolor, detpolcolor, detlabels = \
            generate_hex(7, 5.0, "qu", 15.0)
        outfile = os.path.join(self.outdir, "out_test_vis_hex_small.png")
        sfp.plot_focalplane(detquats, 6.0, 6.0, outfile,
        	fwhm=detfwhm, facecolor=detcolor, polcolor=detpolcolor,
            labels=detlabels)
        return


    def test_vis_hex_small_rad(self):
        dets, detquats, detfwhm, detcolor, detpolcolor, detlabels = \
            generate_hex(7, 5.0, "radial", 15.0)
        outfile = os.path.join(self.outdir, "out_test_vis_hex_small_rad.png")
        sfp.plot_focalplane(detquats, 6.0, 6.0, outfile,
        	fwhm=detfwhm, facecolor=detcolor, polcolor=detpolcolor,
            labels=detlabels)
        return


    def test_vis_hex_medium(self):
        dets, detquats, detfwhm, detcolor, detpolcolor, detlabels = \
            generate_hex(91, 5.0, "qu", 10.0)
        outfile = os.path.join(self.outdir, "out_test_vis_hex_medium.png")
        sfp.plot_focalplane(detquats, 6.0, 6.0, outfile,
        	fwhm=detfwhm, facecolor=detcolor, polcolor=detpolcolor,
            labels=detlabels)
        return


    def test_vis_hex_large(self):
        dets, detquats, detfwhm, detcolor, detpolcolor, detlabels = \
            generate_hex(217, 5.0, "qu", 5.0)
        outfile = os.path.join(self.outdir, "out_test_vis_hex_large.png")
        sfp.plot_focalplane(detquats, 6.0, 6.0, outfile,
        	fwhm=detfwhm, facecolor=detcolor, polcolor=detpolcolor,
            labels=detlabels)
        return


    def test_vis_rhombus(self):
        sixty = np.pi/3.0
        thirty = np.pi/6.0
        rtthree = np.sqrt(3.0)

        rdim = 8
        rpix = rdim**2

        hexwidth = 5.0
        rwidth = hexwidth / rtthree

        # angular separation of rhombi
        margin = 0.60 * hexwidth

        centers = [
            np.array([0.5*margin, 0.0, 0.0]),
            np.array([-0.5*np.cos(sixty)*margin, 0.5*np.sin(sixty)*margin,
                120.0]),
            np.array([-0.5*np.cos(sixty)*margin, -0.5*np.sin(sixty)*margin,
                240.0])
        ]

        cquats = sfp.cartesian_to_quat(centers)

        dets = dict()
        detquats = dict()
        detfwhm = dict()
        detcolor = dict()
        detpolcolor = dict()
        detlabels = dict()
        for w, c in enumerate(cquats):
            wdets, wdetquats, wdetfwhm, wdetcolor, wdetpolcolor, wdetlabels = \
                generate_rhombus(rpix, rwidth, 7.0, "{}".format(w), c)
            dets.update(wdets)
            detquats.update(wdetquats)
            detfwhm.update(wdetfwhm)
            detcolor.update(wdetcolor)
            detpolcolor.update(wdetpolcolor)
            detlabels.update(wdetlabels)

        outfile = os.path.join(self.outdir, "out_test_vis_rhombus.png")
        sfp.plot_focalplane(detquats, 1.2*hexwidth, 1.2*hexwidth, outfile,
        	fwhm=detfwhm, facecolor=detcolor, polcolor=detpolcolor,
            labels=detlabels)
        return
