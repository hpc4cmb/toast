# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..instrument_coords import quat_to_xieta
from ..instrument_sim import plot_focalplane
from ..observation import default_values as defaults
from ..pixels import PixelData
from ._helpers import close_data, create_healpix_ring_satellite, create_outdir
from .mpi import MPITestCase


class DerivativesWeightsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.nside = 64
        
    def healpix_derivatives(self, I_sky):
        npix = hp.nside2npix(self.nside)
        alm = np.array(hp.map2alm(I_sky), dtype=complex)
        theta, phi = hp.pix2ang(self.nside, np.arange(npix))
        _, dtheta, dphi = hp.alm2map_der1(alm, self.nside) #1/sin(theta) dphi
        _, d2theta, dphidtheta = hp.alm2map_der1(hp.map2alm(dtheta), self.nside)
        _, dthetadphi, d2phi = hp.alm2map_der1(hp.map2alm(dphi), self.nside) #1/sin^2(theta) d2phi
        return dtheta, dphi, d2theta, dphidtheta, d2phi

    def create_sky(self, data, pixels, I_sky):

        # Build the pixel distribution
        build_dist = ops.BuildPixelDistribution(
            pixel_pointing=pixels,
        )
        build_dist.apply(data)
        ops.Delete(detdata=[pixels.pixels]).apply(data)

        # Create a fake sky with fixed I/dI/d2I values at all pixels.  Just
        # one submap on all processes.
        dist = data["pixel_dist"]
        npix = 12*self.nside*self.nside
        sm, si = dist.global_pixel_to_submap(np.arange(npix))
        pix_data = PixelData(dist, np.float64, n_value=6, units=u.K)
        dtheta, dphi, d2theta, dphidtheta, d2phi = self.healpix_derivatives(I_sky)
        #Forgive me for the worse distribution job ever
        pix_data.data[sm, si, 0] = I_sky
        pix_data.data[sm, si, 1] = dtheta
        pix_data.data[sm, si, 2] = dphi
        pix_data.data[sm, si, 3] = d2theta
        pix_data.data[sm, si, 4] = dphidtheta
        pix_data.data[sm, si, 5] = d2phi
        return pix_data
    
    def test_generation(self):
        I_sky = np.ones(12*self.nside*self.nside)
        #Create sky map and its derivatives and distribute it
        
        # Create a telescope with 2 boresight pixels per process, so that
        # each process can compare 4 detector orientations.  The boresight
        # pointing will be centered on each pixel exactly once.
        data = create_healpix_ring_satellite(
            self.comm, pix_per_process=2, nside=self.nside
        )

        #Add a detector calibration dictionary
        for ob in data.obs:
            detcal = dict()
            detdx = dict()
            detdy = dict()
            detdsigma = dict()
            detdc = dict()
            detdp = dict()
            for det in ob.local_detectors:
                detcal[det] = 1.0
                detdx[det] = 0.01
                detdy[det] = -0.01
                detdsigma[det] = 0.03
                detdc[det] = 0.995
                detdp[det] = 1.05
            ob["det_cal"] = detcal
            ob["det_dx"] = detdx
            ob["det_dy"] = detdy
            ob["det_dsigma"] = detdsigma
            ob["det_dc"] = detdc
            ob["det_dp"] = detdp

        # Pointing operator
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            detector_pointing=detpointing,
        )
        # Create the input sky and distribute it
        data["sky"] = self.create_sky(data, pixels, I_sky)
        
        weights = ops.DerivativesWeights(
            mode="d2I",
            detector_pointing=detpointing,
            fp_gamma="gamma",
            cal="det_cal",
            dx="det_dx",
            dy="det_dy",
            dsigma="det_dsigma",
            dp="det_dp",
            dc="det_dc",
            IAU=False,
        )
        
        scan_map = ops.ScanMap(
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="sky",
        )

        # Scan map into timestream
        pipe = ops.Pipeline(operators=[pixels, weights, scan_map])
        pipe.apply(data)
        
        ops.Delete(detdata=[pixels.pixels, weights.weights]).apply(data)
        close_data(data)
        