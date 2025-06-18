# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..observation import default_values as defaults
from ..vis import set_matplotlib_backend
from .helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class SSSTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_sss_nopol(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel, quats="quats_azel"
        )

        # Stokes weights
        stokes_weights_azel = ops.StokesWeights(
            detector_pointing=detpointing_azel,
            mode="I",
        )

        # Simulate
        sss = ops.SimScanSynchronousSignal(
            det_data=defaults.det_data,
            detector_pointing=detpointing_azel,
            stokes_weights=stokes_weights_azel,
        )
        sss.apply(data)

        # Minimal testing
        for ob in data.obs:
            dets = ob.select_local_detectors(flagmask=defaults.det_mask_invalid)
            fp = ob.telescope.focalplane
            signal = ob.detdata[defaults.det_data]
            ndet = len(dets)
            for idet1 in range(ndet):
                det1 = dets[idet1]
                pix1 = fp[det1]["pixel"]
                sig1 = signal[det1]
                # Signal must be non-zero
                assert np.std(sig1) != 0
                for idet2 in range(idet1 + 1, ndet):
                    det2 = dets[idet2]
                    pix2 = fp[det2]["pixel"]
                    sig2 = signal[det2]
                    if pix1 == pix2:
                        # Detectors in the same pixel see the same SSS
                        assert np.std(sig1 - sig2) < np.std(sig1) * 1e-10
                    else:
                        # Detectors in different pixels see different SSS
                        assert np.std(sig1 - sig2) > np.std(sig1) * 1e-10

        close_data(data)

    def test_sss_pol(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel, quats="quats_azel"
        )

        # Stokes weights
        stokes_weights_azel = ops.StokesWeights(
            detector_pointing=detpointing_azel,
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
        )

        # Simulate
        sss = ops.SimScanSynchronousSignal(
            detector_pointing=detpointing_azel,
            stokes_weights=stokes_weights_azel,
        )
        sss.apply(data)

        # Minimal testing
        for ob in data.obs:
            dets = ob.select_local_detectors(flagmask=defaults.det_mask_invalid)
            fp = ob.telescope.focalplane
            signal = ob.detdata[defaults.det_data]
            ndet = len(dets)
            for idet1 in range(ndet):
                det1 = dets[idet1]
                pix1 = fp[det1]["pixel"]
                sig1 = signal[det1]
                # Signal must be non-zero
                assert np.std(sig1) != 0
                for idet2 in range(idet1 + 1, ndet):
                    det2 = dets[idet2]
                    pix2 = fp[det2]["pixel"]
                    sig2 = signal[det2]
                    if pix1 == pix2:
                        # Detectors in the same pixel see the same I but different Q/U
                        assert np.std(sig1 - sig2) < np.std(sig1)
                        assert np.std(sig1 - sig2) > np.std(sig1) * 0.01
                    else:
                        # Detectors in different pixels see different SSS
                        assert np.std(sig1 - sig2) > np.std(sig1) * 1e-10

        close_data(data)
