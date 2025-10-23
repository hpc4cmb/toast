# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from .helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class T2PFilterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)
        self.shared_flag_mask = 1
        if (
            ("CONDA_BUILD" in os.environ)
            or ("CIBUILDWHEEL" in os.environ)
            or ("CI" in os.environ)
        ):
            self.make_plots = False
        else:
            self.make_plots = True

    def test_t2pfilter(self):
        # Create a fake ground observations set for testing
        data = create_ground_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model and save for comparison
        sim_noise = ops.SimNoise(noise_model="noise_model", out=defaults.det_data)
        sim_noise.apply(data)

        # Demodulate

        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
            shared_flag_mask=0,
            quats="quats_radec",
        )

        weights_radec = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_radec,
            weights="weights_radec",
        )

        demod_radec = ops.Demodulate(
            stokes_weights=weights_radec,
            in_place=True,
        )
        demod_radec.apply(data)

        # Replace polarized signal with T2P

        for ob in data.obs:
            for det in ob.local_detectors:
                if det.startswith("demod0"):
                    qdet = det.replace("demod0", "demod4r")
                    udet = det.replace("demod0", "demod4i")
                    isig = ob.detdata[defaults.det_data][det]
                    qsig = ob.detdata[defaults.det_data][qdet]
                    usig = ob.detdata[defaults.det_data][udet]
                    qsig[:] = np.random.randn() + np.random.randn() * isig
                    usig[:] = np.random.randn() + np.random.randn() * isig

        # Filter

        t2pfilter = ops.T2PFilter(view="scanning")
        t2pfilter.apply(data)

        # Verify
        for ob in data.obs:
            for det in ob.local_detectors:
                if det.startswith("demod0"):
                    qdet = det.replace("demod0", "demod4r")
                    udet = det.replace("demod0", "demod4i")
                    isig = ob.detdata[defaults.det_data][det]
                    qsig = ob.detdata[defaults.det_data][qdet]
                    usig = ob.detdata[defaults.det_data][udet]
                    qflag = ob.detdata[defaults.det_flags][qdet]
                    uflag = ob.detdata[defaults.det_flags][udet]
                    qgood = qflag & t2pfilter.filter_flag_mask == 0
                    ugood = uflag & t2pfilter.filter_flag_mask == 0
                    qrms = np.std(qsig[qgood])
                    urms = np.std(usig[ugood])
                    assert qrms < 1e-10
                    assert urms < 1e-10

        close_data(data)
