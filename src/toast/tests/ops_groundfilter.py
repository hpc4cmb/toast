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


class GroundFilterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)
        self.shared_flag_mask = 1

    def test_groundfilter(self):
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

        # Add scan-synchronous signal to the data
        for ob in data.obs:
            az = ob.shared[defaults.azimuth].data * 100
            for det in ob.local_detectors:
                ob.detdata[defaults.det_data][det] += az

        ops.Copy(detdata=[(defaults.det_data, "signal_copy")]).apply(data)

        # Filter

        groundfilter = ops.GroundFilter(
            trend_order=0,
            filter_order=1,
            detrend=True,
            split_template=False,
            det_data=defaults.det_data,
            det_flags=defaults.det_flags,
            det_flag_mask=defaults.det_mask_invalid,
            shared_flags=defaults.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            view=None,
        )
        groundfilter.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                flags = ob.shared[defaults.shared_flags].data & self.shared_flag_mask
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                orig_signal = ob.detdata["signal_orig"][det]
                old_signal = ob.detdata["signal_copy"][det]
                new_signal = ob.detdata[defaults.det_data][det]
                # Check that the filtered signal is cleaner than the input signal
                self.assertTrue(
                    np.std(new_signal[good]) < 0.1 * np.std(old_signal[good])
                )
                # Check that the flagged samples were also cleaned and not,
                # for example, set to zero
                self.assertTrue(
                    np.std(new_signal - orig_signal) < 0.1 * np.std(new_signal)
                )
        close_data(data)

    def test_groundfilter_split(self):
        # Create a fake satellite data set for testing
        data = create_ground_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out=defaults.det_data)
        sim_noise.apply(data)

        # Make fake flags
        fake_flags(data)

        rms = dict()
        for ob in data.obs:
            shared_flags = ob.shared[defaults.shared_flags].data
            rightgoing = (shared_flags & defaults.scan_leftright) != 0
            leftgoing = (shared_flags & defaults.scan_rightleft) != 0
            az = ob.shared[defaults.azimuth].data * 100
            rms[ob.name] = dict()
            for det in ob.local_detectors:
                flags = ob.shared[defaults.shared_flags].data & self.shared_flag_mask
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                # Add scan-synchronous signal to the data
                ob.detdata[defaults.det_data][det][leftgoing] += az[leftgoing]
                ob.detdata[defaults.det_data][det][rightgoing] -= az[rightgoing]
                rms[ob.name][det] = np.std(ob.detdata[defaults.det_data][det][good])

        # Filter

        groundfilter = ops.GroundFilter(
            trend_order=0,
            filter_order=1,
            detrend=True,
            split_template=True,
            det_data=defaults.det_data,
            det_flags=defaults.det_flags,
            det_flag_mask=255,
            shared_flags=defaults.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            view=None,
        )
        groundfilter.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                flags = ob.shared[defaults.shared_flags].data & self.shared_flag_mask
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                check_rms = np.std(ob.detdata[defaults.det_data][det][good])
                # print(f"check_rms = {check_rms}, det rms = {rms[ob.name][det]}")
                self.assertTrue(check_rms < 0.1 * rms[ob.name][det])
        close_data(data)
