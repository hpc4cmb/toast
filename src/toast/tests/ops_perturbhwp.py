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
from ..vis import set_matplotlib_backend
from .helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class PerturbHWPTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_perturbhwp(self):
        # Create fake observing of a small patch
        data = create_ground_data(
            self.comm,
            sample_rate=10.0 * u.Hz,
            hwp_rpm=1.0,
        )

        # Copy original HWP to a different field for later comparison
        ops.Copy(
            shared=[
                (defaults.hwp_angle, "hwp_orig"),
            ]
        ).apply(data)

        perturb = ops.PerturbHWP(
            drift_sigma=0.1 / u.h,
            time_sigma=1e-3 * u.s,
            realization=1,
        )
        perturb.apply(data)

        for ob in data.obs:
            times = ob.shared[defaults.times].data
            perturbed = np.unwrap(ob.shared[defaults.hwp_angle].data)
            orig = np.unwrap(ob.shared["hwp_orig"].data)
            rms = np.std(perturbed - orig)

            if data.comm.group_rank == 0:
                set_matplotlib_backend()
                import matplotlib.pyplot as plt

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

                outfile = os.path.join(self.outdir, f"HWP_comparison_{ob.name}.png")
                fig.savefig(outfile)

            self.assertTrue(
                rms > 1e-6,
                msg="HWP angle does not change enough when perturbed",
            )

        close_data(data)

    def test_perturbhwp_stepped(self):
        # Create fake observing of a small patch
        data = create_ground_data(
            self.comm,
            sample_rate=10.0 * u.Hz,
            hwp_rpm=1.0,
        )

        for ob in data.obs:
            # Only one process in column communicator should
            # set shared objects with a non-None value.
            new_val = None
            if ob.comm_col_rank == 0:
                new_val = np.round(np.linspace(0, 10, ob.n_local_samples)) * np.pi / 8
            ob.shared[defaults.hwp_angle].set(new_val, offset=(0,), fromrank=0)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        perturb = ops.PerturbHWP(
            drift_sigma=0.1 / u.h,
            time_sigma=1e-3 * u.s,
            realization=1,
        )

        with self.assertRaises(ValueError):
            perturb.apply(data)

        close_data(data)
