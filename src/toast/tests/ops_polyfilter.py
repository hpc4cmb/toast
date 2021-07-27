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

from ._helpers import create_outdir, create_satellite_data, create_fake_sky, fake_flags


class PolyFilterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_polyfilter(self):

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

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
            rms[ob.name] = dict()
            for det in ob.local_detectors:
                flags = np.array(ob.shared["flags"])
                flags |= ob.detdata["flags"][det]
                good = (flags == 0)
                # Add an offset to the data
                ob.detdata["signal"][det] += 500.0
                rms[ob.name][det] = np.std(ob.detdata["signal"][det][good])

        # Filter

        polyfilter = ops.PolyFilter(
            order=0,
            det_data="signal",
            det_flags="flags",
            det_flag_mask=255,
            shared_flags="flags",
            shared_flag_mask=255,
            view=None,
        )
        polyfilter.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                flags = np.array(ob.shared["flags"])
                flags |= ob.detdata["flags"][det]
                good = (flags == 0)
                check_rms = np.std(ob.detdata["signal"][det][good])
                # print(f"check_rms = {check_rms}, det rms = {rms[ob.name][det]}")
                self.assertTrue(0.9 * check_rms < rms[ob.name][det])

        del data
        return

    def test_polyfilter2D(self):

        testdir = os.path.join(self.outdir, "test_polyfilter2D")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm, pixel_per_process=2)

        # Add wafer IDs for filtering
        for obs in data.obs:
            fp = obs.telescope.focalplane.detector_data
            ndet = len(fp)
            fp.add_column(Column(name="wafer", length=ndet, dtype=int))
            for idet, det in enumerate(fp["name"]):
                fp[idet]["wafer"] = det.endswith("A")

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

        # Make fake flags
        fake_flags(data)

        rms = dict()
        offset = None
        for ob in data.obs:
            rms[ob.name] = dict()
            times = ob.shared["times"].data
            if offset is None:
                offset = times[0]
            for det in ob.local_detectors:
                flags = np.array(ob.shared["flags"])
                flags |= ob.detdata["flags"][det]
                good = (flags == 0)
                wafer = obs.telescope.focalplane[det]["wafer"]
                # Replace signal with time stamps to get a common mode
                ob.detdata["signal"][det] = times - offset + wafer * 1000
                rms[ob.name][det] = np.std(ob.detdata["signal"][det][good])

        # Plot unfiltered TOD

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=[18, 12])
            ax = fig.add_subplot(1, 1, 1)
            ob = data.obs[0]
            for idet, det in enumerate(ob.local_detectors):
                flags = np.array(ob.shared["flags"])
                flags |= ob.detdata["flags"][det]
                good = (flags == 0)
                signal = ob.detdata["signal"][det]
                x = np.arange(signal.size)
                ax.plot(x[good], signal[good], "-", label=det)

        # Filter

        polyfilter = ops.PolyFilter2D(
            order=0,
            det_data="signal",
            det_flags="flags",
            det_flag_mask=255,
            shared_flags="flags",
            shared_flag_mask=255,
            view=None,
            focalplane_key="wafer",
        )
        polyfilter.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                flags = np.array(ob.shared["flags"])
                flags |= ob.detdata["flags"][det]
                good = (flags == 0)
                check_rms = np.std(ob.detdata["signal"][det][good])
                # print(f"check_rms = {check_rms}, det rms = {rms[ob.name][det]}")
                self.assertTrue(check_rms < 1e-3 * rms[ob.name][det])

        # Plot filtered TOD

        if data.comm.world_rank == 0:
            for idet, det in enumerate(ob.local_detectors):
                flags = np.array(ob.shared["flags"])
                flags |= ob.detdata["flags"][det]
                good = (flags == 0)
                signal = ob.detdata["signal"][det]
                x = np.arange(signal.size)
                ax.plot(x[good], signal[good], ".", label=det)
            outfile = os.path.join(testdir, "2Dfiltered_tod.png")
            fig.savefig(outfile)

        del data
        return

    def test_common_mode_filter(self):

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

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

        # Make fake flags
        fake_flags(data)

        rms = dict()
        for ob in data.obs:
            rms[ob.name] = dict()
            times = ob.shared["times"]
            for det in ob.local_detectors:
                flags = np.array(ob.shared["flags"])
                flags |= ob.detdata["flags"][det]
                good = (flags == 0)
                # Replace signal with time stamps to get a common mode
                ob.detdata["signal"][det] = times
                rms[ob.name][det] = np.std(ob.detdata["signal"][det][good])

        # Filter

        common_filter = ops.CommonModeFilter(
            det_data="signal",
            det_flags="flags",
            det_flag_mask=255,
            shared_flags="flags",
            shared_flag_mask=255,
            view=None,
        )
        common_filter.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                flags = np.array(ob.shared["flags"])
                flags |= ob.detdata["flags"][det]
                good = (flags == 0)
                check_rms = np.std(ob.detdata["signal"][det][good])
                # print(f"check_rms = {check_rms}, det rms = {rms[ob.name][det]}")
                self.assertTrue(check_rms < 1e-3 * rms[ob.name][det])

        del data
        return
