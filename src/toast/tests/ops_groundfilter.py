# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from astropy import units as u
from astropy.table import Column

from .mpi import MPITestCase

from ..noise import Noise

from .. import ops as ops

from ..vis import set_matplotlib_backend

from ..pixels import PixelDistribution, PixelData

from ._helpers import create_outdir, create_ground_data, fake_flags


"""
from .mpi import MPITestCase

import os

import numpy as np

from ..tod import AnalyticNoise, OpSimNoise
from ..todmap import OpGroundFilter, TODGround

from ._helpers import create_outdir, create_distdata, boresight_focalplane
"""


class GroundFilterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)
        self.shared_flag_mask = 1

    def test_groundfilter(self):

        # Create a fake satellite data set for testing
        data = create_ground_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pointing = ops.PointingHealpix(
            nside=64,
            mode="IQU",
            hwp_angle="hwp_angle",
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pointing.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out="signal")
        sim_noise.apply(data)

        # Make fake flags
        fake_flags(data)

        rms = dict()
        for ob in data.obs:
            az = ob.shared["azimuth"].data * 100
            rms[ob.name] = dict()
            for det in ob.local_detectors:
                flags = ob.shared["flags"].data & self.shared_flag_mask
                flags |= ob.detdata["flags"][det]
                good = flags == 0
                # Add scan-synchronous signal to the data
                ob.detdata["signal"][det] += az
                rms[ob.name][det] = np.std(ob.detdata["signal"][det][good])

        # Filter

        groundfilter = ops.GroundFilter(
            trend_order=0,
            filter_order=1,
            detrend=True,
            split_template=False,
            det_data="signal",
            det_flags="flags",
            det_flag_mask=255,
            shared_flags="flags",
            shared_flag_mask=self.shared_flag_mask,
            view=None,
        )
        groundfilter.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                flags = ob.shared["flags"].data & self.shared_flag_mask
                flags |= ob.detdata["flags"][det]
                good = flags == 0
                check_rms = np.std(ob.detdata["signal"][det][good])
                # print(f"check_rms = {check_rms}, det rms = {rms[ob.name][det]}")
                self.assertTrue(check_rms < 0.1 * rms[ob.name][det])

        del data
        return

    def test_groundfilter_split(self):

        # Create a fake satellite data set for testing
        data = create_ground_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pointing = ops.PointingHealpix(
            nside=64,
            mode="IQU",
            hwp_angle="hwp_angle",
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pointing.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out="signal")
        sim_noise.apply(data)

        # Make fake flags
        fake_flags(data)

        rms = dict()
        for ob in data.obs:
            shared_flags = ob.shared["flags"].data
            rightgoing = (shared_flags & 2) != 0
            leftgoing = (shared_flags & 4) != 0
            az = ob.shared["azimuth"].data * 100
            rms[ob.name] = dict()
            for det in ob.local_detectors:
                flags = ob.shared["flags"].data & self.shared_flag_mask
                flags |= ob.detdata["flags"][det]
                good = flags == 0
                # Add scan-synchronous signal to the data
                ob.detdata["signal"][det][leftgoing] += az[leftgoing]
                ob.detdata["signal"][det][rightgoing] -= az[rightgoing]
                rms[ob.name][det] = np.std(ob.detdata["signal"][det][good])

        # Filter

        groundfilter = ops.GroundFilter(
            trend_order=0,
            filter_order=1,
            detrend=True,
            split_template=True,
            det_data="signal",
            det_flags="flags",
            det_flag_mask=255,
            shared_flags="flags",
            shared_flag_mask=self.shared_flag_mask,
            view=None,
        )
        groundfilter.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                flags = ob.shared["flags"].data & self.shared_flag_mask
                flags |= ob.detdata["flags"][det]
                good = flags == 0
                check_rms = np.std(ob.detdata["signal"][det][good])
                # print(f"check_rms = {check_rms}, det rms = {rms[ob.name][det]}")
                self.assertTrue(check_rms < 0.1 * rms[ob.name][det])

        del data
        return
