# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u
from astropy.table import Column

from .. import ops as ops
from ..noise import Noise
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..vis import set_matplotlib_backend
from ._helpers import create_ground_data, create_outdir, fake_flags
from .mpi import MPITestCase


class HWPFilterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)
        self.shared_flag_mask = 1

    def test_hwpfilter(self):

        # Create a fake ground data set for testing
        data = create_ground_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out=defaults.det_data)
        sim_noise.apply(data)

        # Make fake flags
        fake_flags(data)

        rms_noise = dict()
        rms_hwpss = dict()
        for ob in data.obs:
            hwp_angle = ob.shared[defaults.hwp_angle].data
            rms_noise[ob.name] = dict()
            rms_hwpss[ob.name] = dict()
            for det in ob.local_detectors:
                flags = ob.shared[defaults.shared_flags].data & self.shared_flag_mask
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                # Add HWP-synchronous signal to the data
                order = 4
                rms_noise[ob.name][det] = np.std(ob.detdata[defaults.det_data][det][good])
                ob.detdata[defaults.det_data][det] += np.cos(order * hwp_angle)
                rms_hwpss[ob.name][det] = np.std(ob.detdata[defaults.det_data][det][good])

        # Filter

        hwpfilter = ops.HWPFilter(
            trend_order=3,
            filter_order=8,
            detrend=True,
        )
        hwpfilter.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                flags = ob.shared[defaults.shared_flags].data & self.shared_flag_mask
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                check_rms = np.std(ob.detdata[defaults.det_data][det][good])
                # print(
                #     f"check_rms = {check_rms}, "
                #     f"rms_hwpss = {rms_hwpss[ob.name][det]}, "
                #     f"rms_noise = {rms_noise[ob.name][det]}"
                # )
                self.assertTrue(check_rms <= rms_noise[ob.name][det])

        data.clear()
        del data
