# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
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
from .helpers import close_data, create_ground_data, create_outdir, fake_flags
from .mpi import MPITestCase


class GroundFilterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)
        self.shared_flag_mask = defaults.shared_mask_invalid

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
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                flags = ob.shared[defaults.shared_flags].data & self.shared_flag_mask
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                orig_signal = ob.detdata["signal_orig"][det]
                old_signal = ob.detdata["signal_copy"][det]
                new_signal = ob.detdata[defaults.det_data][det]
                # Check that the filtered signal is cleaner than the input signal
                orig_rms = np.std(orig_signal[good])
                old_rms = np.std(old_signal[good])
                new_rms = np.std(new_signal[good])
                dof = orig_signal[good].size - 1
                threshold = (1.0 + 1.0 / dof) * orig_rms

                self.assertTrue(new_rms < threshold)
                # Check that the flagged samples were also cleaned and not,
                # for example, set to zero
                self.assertTrue(
                    np.std(new_signal - orig_signal) < 0.1 * np.std(new_signal)
                )
        close_data(data)

    def test_groundfilter_split(self):
        # Create a fake data set for testing
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
            az = ob.shared[defaults.azimuth].data * 100
            rightgoing = np.zeros(az.size, dtype=bool)
            for ival in ob.intervals[defaults.throw_leftright_interval]:
                rightgoing[ival.first : ival.last] = True
            leftgoing = np.zeros(az.size, dtype=bool)
            for ival in ob.intervals[defaults.throw_rightleft_interval]:
                leftgoing[ival.first : ival.last] = True
            rms[ob.name] = dict()
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
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
            det_flag_mask=defaults.det_mask_invalid,
            shared_flags=defaults.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            view=None,
        )
        groundfilter.apply(data)

        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                flags = ob.shared[defaults.shared_flags].data & self.shared_flag_mask
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                check_rms = np.std(ob.detdata[defaults.det_data][det][good])
                # print(f"check_rms = {check_rms}, det rms = {rms[ob.name][det]}")
                self.assertTrue(check_rms < 0.1 * rms[ob.name][det])
        close_data(data)

    def test_groundfilter_split_binned(self):
        # Create a fake data set for testing
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
            az = ob.shared[defaults.azimuth].data * 100
            rightgoing = np.zeros(az.size, dtype=bool)
            for ival in ob.intervals[defaults.throw_leftright_interval]:
                rightgoing[ival.first : ival.last] = True
            leftgoing = np.zeros(az.size, dtype=bool)
            for ival in ob.intervals[defaults.throw_rightleft_interval]:
                leftgoing[ival.first : ival.last] = True
            rms[ob.name] = dict()
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                flags = ob.shared[defaults.shared_flags].data & self.shared_flag_mask
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                # Add scan-synchronous signal to the data
                ob.detdata[defaults.det_data][det][leftgoing] += az[leftgoing]
                ob.detdata[defaults.det_data][det][rightgoing] -= az[rightgoing]
                rms[ob.name][det] = np.std(ob.detdata[defaults.det_data][det][good])

        # Filter

        groundfilter = ops.GroundFilter(
            trend_order=None,
            filter_order=None,
            bin_width=1 * u.deg,
            detrend=True,
            split_template=True,
            det_data=defaults.det_data,
            det_flags=defaults.det_flags,
            det_flag_mask=defaults.det_mask_invalid,
            shared_flags=defaults.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            view=None,
        )
        groundfilter.apply(data)

        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                flags = ob.shared[defaults.shared_flags].data & self.shared_flag_mask
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                check_rms = np.std(ob.detdata[defaults.det_data][det][good])
                # print(f"check_rms = {check_rms}, det rms = {rms[ob.name][det]}")
                self.assertTrue(check_rms < 0.1 * rms[ob.name][det])
        close_data(data)

    def test_redistributed(self):
        # Create a fake data set for testing
        data = create_ground_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out=defaults.det_data)
        sim_noise.apply(data)

        # Redistribute the data.  If there is more than one process in a
        # group, this will render the data incompatible with the operator
        for ob in data.obs:
            ob.redistribute(1, times=defaults.times)

        caught = False
        try:
            groundfilter = ops.GroundFilter()
            groundfilter.apply(data)
        except RuntimeError as e:
            caught = True

        if data.comm.group_size == 1:
            self.assertFalse(caught)
        else:
            self.assertTrue(caught)

        close_data(data)
