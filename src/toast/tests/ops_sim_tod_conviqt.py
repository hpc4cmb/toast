# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from astropy import units as u

import healpy as hp

from .mpi import MPITestCase

from ..vis import set_matplotlib_backend

from .. import qarray as qa

from .. import ops as ops

from ..pixels_io import write_healpix_fits

from ..dipole import dipole

from ._helpers import (
    create_outdir,
    create_healpix_ring_satellite,
    create_fake_sky_alm,
    create_fake_beam_alm,
)


class SimConviqtTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.nside = 64
        self.lmax = 128
        self.mmax = 10
        self.fwhm = 10 * u.degree
        self.fname_sky = os.path.join(self.outdir, "sky_alm.fits")
        self.fname_beam = os.path.join(self.outdir, "beam_alm.fits")

        # Synthetic sky and beam (a_lm expansions)
        slm = create_fake_sky_alm(self.lmax, self.fwhm)
        hp.write_alm(self.fname_sky, slm, lmax=self.lmax)

        blm = create_fake_beam_alm(self.lmax, self.mmax)
        hp.write_alm(self.fname_beam, blm, lmax=self.lmax, mmax_in=self.mmax)

        return

    def test_sim(self):
        # Create a fake scan strategy that hits every pixel once.
        data = create_healpix_ring_satellite(self.comm, nside=self.nside)

        # make a simple pointing matrix
        detpointing = ops.PointingDetectorSimple()

        # Generate timestreams
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file=self.fname_beam,
            dxx=False,
        )
        sim_conviqt.exec(data)

        return
