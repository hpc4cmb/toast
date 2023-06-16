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
from ._helpers import close_data, create_ground_data, create_outdir, fake_flags
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

        ops.Copy(detdata=[(defaults.det_data, "signal_orig")]).apply(data)

        # Make fake flags
        fake_flags(data)

        for ob in data.obs:
            hwp_angle = ob.shared[defaults.hwp_angle].data
            for det in ob.local_detectors:
                # Add HWP-synchronous signal to the data
                order = 4
                ob.detdata[defaults.det_data][det] += np.cos(order * hwp_angle)

        ops.Copy(detdata=[(defaults.det_data, "signal_copy")]).apply(data)

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
                orig_signal = ob.detdata["signal_orig"][det]
                old_signal = ob.detdata["signal_copy"][det]
                new_signal = ob.detdata[defaults.det_data][det]
                # Check that the filtered signal is cleaner than the input signal
                self.assertTrue(np.std(new_signal[good]) < np.std(old_signal[good]))
                # Check that the flagged samples were also cleaned and not,
                # for example, set to zero. Use np.diff() to remove any
                # residual trend
                self.assertTrue(
                    np.std(np.diff(new_signal) - np.diff(orig_signal))
                    < 0.1 * np.std(np.diff(new_signal))
                )

        close_data(data)
