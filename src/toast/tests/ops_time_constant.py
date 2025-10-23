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
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..vis import set_matplotlib_backend
from .helpers import (
    close_data,
    create_outdir,
    create_satellite_data,
)
from .mpi import MPITestCase

XAXIS, YAXIS, ZAXIS = np.eye(3)


class TimeConstantTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        if (
            ("CONDA_BUILD" in os.environ)
            or ("CIBUILDWHEEL" in os.environ)
            or ("CI" in os.environ)
        ):
            self.make_plots = False
        else:
            self.make_plots = True

    def test_phase_shift(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm, flagged_pixels=False)

        # Create impulse delta
        for ob in data.obs:
            nsamp = ob.n_local_samples
            mid = nsamp // 2
            nramp = 50
            ramp = np.arange(nramp) / nramp
            for det in ob.local_detectors:
                sig = ob.detdata[defaults.det_data][det]
                sig[mid - nramp : mid] = ramp
                sig[mid : mid + nramp] = 1 - ramp

        # Convolve
        time_constant = ops.TimeConstant(
            tau=1.0 * u.second,
            det_data="signal",
        )
        time_constant.apply(data)

        for ob in data.obs:
            nsamp = ob.n_local_samples
            mid = nsamp // 2
            for det in ob.local_detectors:
                sig = ob.detdata[defaults.det_data][det]
                peak = np.amax(sig)
                peak_loc = np.argmax(sig)
                self.assertTrue(peak < 1.0)
                self.assertTrue(peak_loc > mid)

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
        if self.make_plots:
            time_constant.debug = self.outdir
        time_constant.deconvolve = False
        time_constant.apply(data)

        # Verify that the signal changed
        for obs in data.obs:
            for det in obs.select_local_detectors(flagmask=defaults.det_mask_invalid):
                slc = slice(100, -100, 1)
                signal0 = obs.detdata["signal0"][det][slc]
                signal = obs.detdata["signal"][det][slc]
                rms = np.std(signal0 - signal) / np.std(signal0)
                self.assertTrue(rms > 0.01)

        # Now deconvolve

        time_constant.deconvolve = True
        time_constant.apply(data)

        # Verify that the signal is restored
        for obs in data.obs:
            if obs.comm.group_rank == 0:
                import matplotlib.pyplot as plt
            for det in obs.select_local_detectors(flagmask=defaults.det_mask_invalid):
                slc = slice(100, -100, 1)
                signal0 = obs.detdata["signal0"][det]
                signal = obs.detdata["signal"][det]
                if obs.comm.group_rank == 0:
                    for prange in [(0, len(signal)), (0, 200)]:
                        pslc = slice(prange[0], prange[1], 1)
                        fig = plt.figure(figsize=(12, 8), dpi=72)
                        ax = fig.add_subplot(2, 1, 1, aspect="auto")
                        ax.plot(
                            obs.shared["times"].data[pslc],
                            signal0[pslc],
                            c="black",
                            label=f"Det {det} Original",
                        )
                        ax.plot(
                            obs.shared["times"].data[pslc],
                            signal[pslc],
                            c="red",
                            label=f"Det {det} After Convolve / Deconvolve",
                        )
                        ax.legend(loc=1)
                        ax = fig.add_subplot(2, 1, 2, aspect="auto")
                        ax.plot(
                            obs.shared["times"].data[pslc],
                            (signal - signal0)[pslc],
                            c="blue",
                            label=f"Det {det} Processed - Original",
                        )
                        # ax.set_ylim(-0.1, 0.1)
                        ax.legend(loc=1)
                        plt.title(f"Observation {obs.name} Detector {det}")
                        savefile = os.path.join(
                            self.outdir,
                            f"out_{obs.name}_{det}_{prange[0]}-{prange[1]}.pdf",
                        )
                        plt.savefig(savefile)
                        plt.close()

                rms = np.std(signal0[slc] - signal[slc]) / np.std(signal0[slc])
                self.assertTrue(rms < 0.05)

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
            det_data="signal0",
        )
        time_constant.apply(data)

        # Convolve with error

        time_constant = ops.TimeConstant(
            tau=1 * u.ms,
            tau_sigma=0.1,
            det_data="signal",
        )
        time_constant.apply(data)

        # Verify that the signal is different
        for obs in data.obs:
            for det in obs.select_local_detectors(flagmask=defaults.det_mask_invalid):
                signal0 = obs.detdata["signal0"][det]
                signal = obs.detdata["signal"][det]
                rms = np.std(signal - signal0) / np.std(signal0)
                self.assertTrue(rms < 0.2)
                self.assertTrue(rms > 1e-8)

        close_data(data)
