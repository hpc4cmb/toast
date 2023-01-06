# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..noise import Noise
from ..observation import default_values as defaults
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


class MadamTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_madam_det_out(self):
        if not ops.madam.available():
            print("libmadam not available, skipping tests")
            return

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            detector_pointing=detpointing,
            create_dist="pixel_dist",
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
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
        sim_noise = ops.SimNoise(noise_model="noise_model", out=defaults.det_data)
        sim_noise.apply(data)

        # Make fake flags
        fake_flags(data)

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
                flags = np.array(ob.shared[defaults.shared_flags])
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                # Add an offset to the data
                ob.detdata[defaults.det_data][det] += 500.0
                rms[ob.name][det] = np.std(ob.detdata[defaults.det_data][det][good])

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
        pars["nside_map"] = pixels.nside
        pars["nside_cross"] = pixels.nside
        pars["nside_submap"] = min(8, pixels.nside)
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
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            det_out="destriped",
            noise_model="noise_model",
            copy_groups=2,
            purge_det_data=False,
            restore_det_data=False,
            shared_flags=defaults.shared_flags,
            shared_flag_mask=1,
            det_flags=defaults.det_flags,
            det_flag_mask=1,
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
                flags = np.array(ob.shared[defaults.shared_flags])
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                check_rms = np.std(ob.detdata["destriped"][det][good])
                # print(f"check_rms = {check_rms}, det rms = {rms[ob.name][det]}")
                self.assertTrue(0.9 * check_rms < rms[ob.name][det])

        close_data(data)
