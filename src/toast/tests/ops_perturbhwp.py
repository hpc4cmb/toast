# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..observation import default_values as defaults
from ..pixels import PixelData
from ..pixels_io_healpix import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class PerturbHWPTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_perturbhwp(self):
        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        times = data.obs[0].shared[defaults.times].data.copy()
        orig = data.obs[0].shared[defaults.hwp_angle].data.copy()

        perturb = ops.PerturbHWP(
            drift_sigma=0.1 / u.h,
            time_sigma=1e-3 * u.s,
            realization=1,
        )
        perturb.apply(data)

        perturbed = data.obs[0].shared[defaults.hwp_angle].data.copy()
        rms = np.std(perturbed - orig)

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            orig = np.unwrap(orig)
            perturbed = np.unwrap(perturbed)
            rms = np.std(perturbed - orig)

            fig = plt.figure(figsize=[18, 12])
            nrow, ncol = 2, 3

            ax = fig.add_subplot(nrow, ncol, 1)
            ax.plot(times, orig, label="Ideal HWP")
            ax.plot(times, perturbed, label="Perturbed HWP")
            ax.legend(loc="best")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("HWP angle [rad]")

            ax = fig.add_subplot(nrow, ncol, 2)
            resid = perturbed - orig
            ax.plot(times, resid, label="Diff")
            ax.legend(loc="best")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("HWP angle [rad]")

            ax = fig.add_subplot(nrow, ncol, 3)
            rate = data.obs[0].telescope.focalplane.sample_rate
            psd = np.abs(np.fft.rfft(resid))
            freq = np.fft.rfftfreq(resid.size, 1 / rate)
            ax.loglog(psd, label="PSD Diff")
            ax.legend(loc="best")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("PSD [Rad / Hz^1/2]")

            outfile = os.path.join(self.outdir, "HWP_comparison.png")
            fig.savefig(outfile)

        assert rms > 1e-6, "HWP angle does not change enough when perturbed"

        close_data(data)
