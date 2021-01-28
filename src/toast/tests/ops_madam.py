# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from astropy import units as u

from .mpi import MPITestCase

from ..noise import Noise

from .. import ops as ops

from ..vis import set_matplotlib_backend

from ..pixels import PixelDistribution, PixelData

from ._helpers import create_outdir, create_satellite_data, create_fake_sky


class MadamTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_madam_det_out(self):
        if not ops.Madam.available:
            print("libmadam not available, skipping tests")
            return

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        pointing = ops.PointingHealpix(
            nside=64, mode="IQU", hwp_angle="hwp_angle", create_dist="pixel_dist"
        )
        pointing.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # if data.comm.world_rank == 0:
        #     set_matplotlib_backend()
        #     import matplotlib.pyplot as plt
        #
        #     ob = data.obs[0]
        #     det = ob.local_detectors[0]
        #     xdata = ob.shared["times"].data
        #     ydata = ob.detdata["signal"][det]
        #
        #     fig = plt.figure(figsize=(12, 8), dpi=72)
        #     ax = fig.add_subplot(1, 1, 1, aspect="auto")
        #     ax.plot(
        #         xdata,
        #         ydata,
        #         marker="o",
        #         c="red",
        #         label="{}, {}".format(ob.name, det),
        #     )
        #     # cur_ylim = ax.get_ylim()
        #     # ax.set_ylim([0.001 * (nse.NET(det) ** 2), 10.0 * cur_ylim[1]])
        #     ax.legend(loc=1)
        #     plt.title("Sky Signal")
        #     savefile = os.path.join(
        #         self.outdir, "signal_sky_{}_{}.pdf".format(ob.name, det)
        #     )
        #     plt.savefig(savefile)
        #     plt.close()

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out="signal")
        sim_noise.apply(data)

        # if data.comm.world_rank == 0:
        #     set_matplotlib_backend()
        #     import matplotlib.pyplot as plt
        #
        #     ob = data.obs[0]
        #     det = ob.local_detectors[0]
        #     xdata = ob.shared["times"].data
        #     ydata = ob.detdata["signal"][det]
        #
        #     fig = plt.figure(figsize=(12, 8), dpi=72)
        #     ax = fig.add_subplot(1, 1, 1, aspect="auto")
        #     ax.plot(
        #         xdata,
        #         ydata,
        #         marker="o",
        #         c="red",
        #         label="{}, {}".format(ob.name, det),
        #     )
        #     # cur_ylim = ax.get_ylim()
        #     # ax.set_ylim([0.001 * (nse.NET(det) ** 2), 10.0 * cur_ylim[1]])
        #     ax.legend(loc=1)
        #     plt.title("Sky + Noise Signal")
        #     savefile = os.path.join(
        #         self.outdir, "signal_sky-noise_{}_{}.pdf".format(ob.name, det)
        #     )
        #     plt.savefig(savefile)
        #     plt.close()

        # Compute timestream rms

        rms = dict()
        for ob in data.obs:
            rms[ob.name] = dict()
            for det in ob.local_detectors:
                # Add an offset to the data
                ob.detdata["signal"][det] += 500.0
                rms[ob.name][det] = np.std(ob.detdata["signal"][det])

        # if data.comm.world_rank == 0:
        #     set_matplotlib_backend()
        #     import matplotlib.pyplot as plt
        #
        #     ob = data.obs[0]
        #     det = ob.local_detectors[0]
        #     xdata = ob.shared["times"].data
        #     ydata = ob.detdata["signal"][det]
        #
        #     fig = plt.figure(figsize=(12, 8), dpi=72)
        #     ax = fig.add_subplot(1, 1, 1, aspect="auto")
        #     ax.plot(
        #         xdata,
        #         ydata,
        #         marker="o",
        #         c="red",
        #         label="{}, {}".format(ob.name, det),
        #     )
        #     # cur_ylim = ax.get_ylim()
        #     # ax.set_ylim([0.001 * (nse.NET(det) ** 2), 10.0 * cur_ylim[1]])
        #     ax.legend(loc=1)
        #     plt.title("Sky + Noise + Offset Signal")
        #     savefile = os.path.join(
        #         self.outdir, "signal_sky-noise-offset_{}_{}.pdf".format(ob.name, det)
        #     )
        #     plt.savefig(savefile)
        #     plt.close()

        # Run madam on this

        # Madam assumes constant sample rate- just get it from the noise model for
        # the first detector.
        sample_rate = data.obs[0]["noise_model"].rate(data.obs[0].local_detectors[0])

        pars = {}
        pars["kfirst"] = "T"
        pars["iter_max"] = 100
        pars["base_first"] = 300.0
        pars["fsample"] = sample_rate
        pars["nside_map"] = pointing.nside
        pars["nside_cross"] = pointing.nside
        pars["nside_submap"] = min(8, pointing.nside)
        pars["write_map"] = "T"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = self.outdir

        # FIXME: add a view here once our test data includes it

        madam = ops.Madam(
            params=pars,
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            pixels_nested=pointing.nest,
            det_out="destriped",
            noise_model="noise_model",
            copy_groups=2,
            purge_det_data=False,
            purge_pointing=True,
        )
        madam.apply(data)

        # if data.comm.world_rank == 0:
        #     set_matplotlib_backend()
        #     import matplotlib.pyplot as plt
        #
        #     ob = data.obs[0]
        #     det = ob.local_detectors[0]
        #     xdata = ob.shared["times"].data
        #     ydata = ob.detdata["destriped"][det]
        #
        #     fig = plt.figure(figsize=(12, 8), dpi=72)
        #     ax = fig.add_subplot(1, 1, 1, aspect="auto")
        #     ax.plot(
        #         xdata,
        #         ydata,
        #         marker="o",
        #         c="red",
        #         label="{}, {}".format(ob.name, det),
        #     )
        #     # cur_ylim = ax.get_ylim()
        #     # ax.set_ylim([0.001 * (nse.NET(det) ** 2), 10.0 * cur_ylim[1]])
        #     ax.legend(loc=1)
        #     plt.title("Destriped Signal")
        #     savefile = os.path.join(
        #         self.outdir, "signal_destriped_{}_{}.pdf".format(ob.name, det)
        #     )
        #     plt.savefig(savefile)
        #     plt.close()

        for ob in data.obs:
            for det in ob.local_detectors:
                check_rms = np.std(ob.detdata["destriped"][det])
                self.assertTrue(0.9 * check_rms < rms[ob.name][det])

        del data
        return
