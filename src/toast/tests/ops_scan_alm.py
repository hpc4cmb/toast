# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from .helpers import (close_data, create_fake_healpix_scanned_tod,
                      create_outdir, create_satellite_data)
from .mpi import MPITestCase


class ScanAlmTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)

    def test_scan_I(self):
        self._test_scan(mode="I")

    def test_scan_QU(self):
        self._test_scan(mode="QU")

    def test_scan_IQU(self):
        self._test_scan(mode="IQU")

    def test_scan_I_det(self):
        self._test_scan(mode="I", focalplane_keys=True)

    def test_scan_QU_det(self):
        self._test_scan(mode="QU", focalplane_keys=True)

    def test_scan_IQU_det(self):
        self._test_scan(mode="IQU", focalplane_keys=True)

    def _test_scan(self, mode, focalplane_keys=False):
        if not ops.scan_alm.ducc_available:
            print("ducc0.totalconvolve is not available skipping tests")
            return

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=256,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        # pixels.apply(data)
        weights_scan = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        # weights.apply(data)

        # Create fake polarized sky signal

        scales = []
        for stokes in "IQU":
            if stokes in mode:
                scales.append(1.0)
            else:
                scales.append(0.0)

        if focalplane_keys:
            pixel_names = np.unique(data.obs[0].telescope.focalplane.detector_data["pixel"])
        else:
            pixel_names = ["none"]

        hpix_file = os.path.join(self.outdir, f"fake_{mode}_{{pixel}}.fits")
        for i, pixel in enumerate(pixel_names):
            map_key = f"fake_map_{pixel}"
            create_fake_healpix_scanned_tod(
                data,
                pixels,
                weights_scan,
                hpix_file.format(pixel=pixel),
                "pixel_dist",
                map_key=map_key,
                fwhm=30.0 * u.degree,
                lmax=3 * pixels.nside,
                I_scale=scales[0],
                Q_scale=scales[1],
                U_scale=scales[2],
                det_data=f"det_data_{pixel}",
            )

        # Expand the input map in spherical harmonics on root process

        alm_file = hpix_file.replace(".fits", ".alm.fits")

        if data.comm.comm_world is None or data.comm.comm_world.rank == 0:
            for i, pixel in enumerate(pixel_names):
                m = hp.read_map(hpix_file.format(pixel=pixel), None)
                nside = hp.get_nside(m)
                lmax = 2 * nside
                alm = hp.map2alm(m, lmax=lmax, iter=0, pol=True)
                if hpix_file == alm_file:
                    raise RuntimeError("Failed to synthesize an alm file name")
                hp.write_alm(alm_file.format(pixel=pixel), alm, out_dtype=np.complex64, lmax=lmax)

        if data.comm.comm_world is not None:
            data.comm.comm_world.Barrier()

        # Scan the alm from the file

        weights_alm = ops.StokesWeights(
            mode=mode,
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
            weights="weights_alm",
        )

        scan_alm = ops.ScanAlm(
            file=alm_file.format(pixel=pixel),
            det_data="interp_data",
            detector_pointing=detpointing,
            stokes_weights=weights_alm,
            focalplane_keys="pixel,psi_pol" if focalplane_keys else None,
        )
        scan_alm.apply(data)

        # Check that the sets of timestreams match.

        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                if focalplane_keys:
                    pixel = data.obs[0].telescope.focalplane[det]["pixel"]
                else:
                    pixel = 'none'
                sig1 = ob.detdata[f"det_data_{pixel}"][det]
                sig2 = ob.detdata["interp_data"][det]
                rms1 = np.std(sig1)
                rms2 = np.std(sig2)
                rmsdiff = np.std(sig1 - sig2)
                if rms1 < 1e-12:
                    # Empty signal, cannot compare relative power
                    continue
                assert np.abs(rms1 / rms2 - 1) < 1e-3
                assert np.abs(rmsdiff / rms1) < 1e-1

        close_data(data)
