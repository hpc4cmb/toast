# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u
from astropy.table import Column

from .. import ops as ops
from .. import qarray as qa
from ..noise import Noise
from ..pixels import PixelData, PixelDistribution
from ..vis import set_matplotlib_backend
from ._helpers import (
    close_data,
    create_fake_sky,
    create_outdir,
    create_satellite_data,
    fake_flags,
)
from .mpi import MPITestCase

XAXIS, YAXIS, ZAXIS = np.eye(3)


class TimeConstantTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_time_constant(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out="signal")
        sim_noise.apply(data)

        # Copy the signal for reference
        ops.Copy(detdata=[("signal", "signal0")]).apply(data)

        # Convolve

        time_constant = ops.TimeConstant(
            tau=1 * u.ms,
            det_data="signal",
        )
        time_constant.deconvolve = False
        time_constant.apply(data)

        # Verify that the signal changed
        for obs in data.obs:
            for det in obs.local_detectors:
                signal0 = obs.detdata["signal0"][det]
                signal = obs.detdata["signal"][det]
                rms = np.std(signal0 - signal) / np.std(signal0)
                assert rms > 0.01

        # Now deconvolve

        time_constant.deconvolve = True
        time_constant.apply(data)

        # Verify that the signal is restored
        for obs in data.obs:
            if obs.comm.group_rank == 0:
                import matplotlib.pyplot as plt
            for det in obs.local_detectors:
                signal0 = obs.detdata["signal0"][det]
                signal = obs.detdata["signal"][det]
                if obs.comm.group_rank == 0:
                    fig = plt.figure(figsize=(12, 8), dpi=72)
                    ax = fig.add_subplot(2, 1, 1, aspect="auto")
                    ax.plot(
                        obs.shared["times"].data,
                        signal0,
                        c="black",
                        label=f"Det {det} Original",
                    )
                    ax.plot(
                        obs.shared["times"].data,
                        signal,
                        c="red",
                        label=f"Det {det} After Convolve / Deconvolve",
                    )
                    ax.legend(loc=1)
                    ax = fig.add_subplot(2, 1, 2, aspect="auto")
                    ax.plot(
                        obs.shared["times"].data,
                        (signal - signal0),
                        c="blue",
                        label=f"Det {det} Processed - Original",
                    )
                    ax.set_ylim(-0.1, 0.1)
                    ax.legend(loc=1)
                    plt.title(f"Observation {obs.name} Detector {det}")
                    savefile = os.path.join(
                        self.outdir,
                        f"out_{obs.name}_{det}.pdf",
                    )
                    plt.savefig(savefile)
                    plt.close()

                rms = np.std(signal0 - signal) / np.std(signal0)
                assert rms < 1e-4

        close_data(data)

    def test_time_constant_error(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out="signal")
        sim_noise.apply(data)

        # Copy the signal for reference
        ops.Copy(detdata=[("signal", "signal0")]).apply(data)

        # Convolve

        time_constant = ops.TimeConstant(
            tau=1 * u.ms,
            det_data="signal",
        )
        time_constant.apply(data)

        # Convolve with error

        time_constant = ops.TimeConstant(
            tau=1 * u.ms,
            tau_sigma=0.01,
            det_data="signal0",
        )
        time_constant.apply(data)

        # Verify that the signal is different
        for obs in data.obs:
            for det in obs.local_detectors:
                signal0 = obs.detdata["signal0"][det]
                signal = obs.detdata["signal"][det]
                rms = np.std(signal0 - signal) / np.std(signal)
                assert rms < 1e-2
                assert rms > 1e-8

        close_data(data)
