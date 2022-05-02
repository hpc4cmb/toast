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
from ._helpers import (
    create_fake_sky,
    create_ground_data,
    create_outdir,
    create_satellite_data,
    fake_flags,
)
from .mpi import MPITestCase

XAXIS, YAXIS, ZAXIS = np.eye(3)


class PolyFilterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_polyfilter(self):

        # Create a fake ground data set for testing
        data = create_ground_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle="hwp_angle",
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

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out="signal")
        sim_noise.apply(data)

        # Make fake flags
        fake_flags(data)

        for ob in data.obs:
            times = np.array(ob.shared[defaults.times])
            for det in ob.local_detectors:
                flags = (
                    np.array(ob.shared[defaults.shared_flags])
                    & defaults.shared_mask_invalid
                ) != 0
                flags |= (
                    ob.detdata[defaults.det_flags][det] & defaults.det_mask_invalid
                ) != 0
                good = np.logical_not(flags)
                signal = ob.detdata[defaults.det_data][det]
                # Replace TOD with a gradient
                signal[:] = times

        ops.Copy(detdata=[(defaults.det_data, "signal_copy")]).apply(data)

        # Filter

        polyfilter = ops.PolyFilter(
            order=1,
            det_data=defaults.det_data,
            det_flags=defaults.det_flags,
            det_flag_mask=defaults.shared_mask_invalid,
            shared_flags=defaults.shared_flags,
            shared_flag_mask=defaults.det_mask_invalid,
            poly_flag_mask=1,
            view="scanning",
        )
        polyfilter.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                flags = (
                    np.array(ob.shared[defaults.shared_flags])
                    & defaults.shared_mask_invalid
                ) != 0
                flags |= (
                    ob.detdata[defaults.det_flags][det] & defaults.det_mask_invalid
                ) != 0
                good = np.logical_not(flags)
                old_signal = ob.detdata["signal_copy"][det]
                new_signal = ob.detdata[defaults.det_data][det]
                old_rms = np.std(old_signal[good])
                new_rms = np.std(new_signal[good])
                # print(
                #     f"new_rms = {new_rms}, old_rms = {old_rms}, new/old = {new_rms / old_rms}"
                # )
                self.assertLess(new_rms / old_rms, 1e-6)
        data.clear()
        del data

    def test_polyfilter2D(self):

        testdir = os.path.join(self.outdir, "test_polyfilter2D")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm, pixel_per_process=4)

        # Add wafer IDs for filtering
        for obs in data.obs:
            fp = obs.telescope.focalplane.detector_data
            ndet = len(fp)
            fp.add_column(Column(name="wafer", length=ndet, dtype=int))
            for idet, det in enumerate(fp["name"]):
                fp[idet]["wafer"] = det.endswith("A")

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
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

        # Add 2D polynomial modes.  The number of modes is larger than the
        # number of detectors to test handling of singular template matrices.
        norder = 5
        coeff = np.arange(norder**2)
        for obs in data.obs:
            for det in obs.local_detectors:
                det_quat = obs.telescope.focalplane[det]["quat"]
                x, y, z = qa.rotate(det_quat, ZAXIS)
                theta, phi = np.arcsin([x, y])
                signal = obs.detdata["signal"][det]
                # signal[:] = 0
                icoeff = 0
                for xorder in range(norder):
                    for yorder in range(norder):
                        signal += coeff[icoeff] * theta**xorder * phi**yorder
                        icoeff += 1
                # Add a different offset depending on the wafer
                wafer = obs.telescope.focalplane[det]["wafer"]
                signal += wafer

        # Make fake flags
        fake_flags(data)

        rms = dict()
        offset = None
        for ob in data.obs:
            good = ob.detdata["flags"].data == 0
            good *= ob.shared["flags"].data == 0
            rms[ob.name] = np.std(ob.detdata["signal"].data[good])

        # Plot unfiltered TOD

        # if data.comm.world_rank == 0:
        #     set_matplotlib_backend()
        #     import matplotlib.pyplot as plt

        #     fig = plt.figure(figsize=[18, 12])
        #     ax = fig.add_subplot(1, 3, 1)
        #     ob = data.obs[0]
        #     for idet, det in enumerate(ob.local_detectors):
        #         flags = np.array(ob.shared["flags"])
        #         flags |= ob.detdata["flags"][det]
        #         good = flags == 0
        #         signal = ob.detdata["signal"][det]
        #         x = np.arange(signal.size)
        #         ax.plot(x, signal, "-", label=f"{det} unfiltered")
        #         ax.plot(x, good, "-", label=f"{det} input good samples")
        #     ax.legend(loc="best")

        # Filter with python implementation.  Make a copy first

        ops.Copy(detdata=[("signal", "pyfilter"), ("flags", "pyflags")]).apply(data)

        polyfilter = ops.PolyFilter2D(
            order=norder - 1,
            det_data="pyfilter",
            det_flags="pyflags",
            det_flag_mask=255,
            shared_flags="flags",
            shared_flag_mask=255,
            view=None,
            focalplane_key="wafer",
        )
        polyfilter.use_python = True
        polyfilter.apply(data)

        # Plot filtered TOD

        # if data.comm.world_rank == 0:
        #     ax = fig.add_subplot(1, 3, 2)
        #     for idet, det in enumerate(ob.local_detectors):
        #         flags = np.array(ob.shared["flags"])
        #         flags |= ob.detdata["flags"][det]
        #         good = flags == 0
        #         signal = ob.detdata["pyfilter"][det]
        #         x = np.arange(signal.size)
        #         ax.plot(x, signal, ".", label=f"{det} filtered")
        #         ax.plot(x, good, "-", label=f"{det} new good samples")
        #     ax.legend(loc="best")

        # Do the same with C++ implementation

        polyfilter.det_data = defaults.det_data
        polyfilter.use_python = False
        polyfilter.apply(data)

        # if data.comm.world_rank == 0:
        #     ax = fig.add_subplot(1, 3, 3)
        #     for idet, det in enumerate(ob.local_detectors):
        #         flags = np.array(ob.shared["flags"])
        #         flags |= ob.detdata["flags"][det]
        #         good = flags == 0
        #         signal = ob.detdata["signal"][det]
        #         x = np.arange(signal.size)
        #         ax.plot(x, signal, ".", label=f"{det} filtered")
        #         ax.plot(x, good, "-", label=f"{det} new good samples")
        #     ax.legend(loc="best")
        #     outfile = os.path.join(testdir, "2Dfiltered_tod.png")
        #     fig.savefig(outfile)

        # Check for consistency
        for ob in data.obs:
            for det in ob.local_detectors:
                # print(f"pyfilter {det} = {ob.detdata['pyfilter'][det]}")
                # print(f"signal {det} = {ob.detdata['signal'][det]}")
                self.assertTrue(
                    np.allclose(ob.detdata["signal"][det], ob.detdata["pyfilter"][det])
                )

        # Check that the filtering reduces RMS
        for ob in data.obs:
            good = ob.detdata["flags"].data == 0
            good *= ob.shared["flags"].data == 0
            check_rms = np.std(ob.detdata["signal"].data[good])
            self.assertLess(check_rms, 1e-3 * rms[ob.name])

        data.clear()
        del data

    def test_common_mode_filter(self):

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle="hwp_angle",
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

        # Make fake flags
        fake_flags(data)

        rms = dict()
        for ob in data.obs:
            rms[ob.name] = dict()
            times = ob.shared["times"]
            for det in ob.local_detectors:
                flags = np.array(ob.shared["flags"])
                flags |= ob.detdata["flags"][det]
                good = flags == 0
                # Replace signal with time stamps to get a common mode
                ob.detdata["signal"][det] = times
                rms[ob.name][det] = np.std(ob.detdata["signal"][det][good])

        # Filter

        common_filter = ops.CommonModeFilter(
            det_data=defaults.det_data,
            det_flags=defaults.det_flags,
            det_flag_mask=255,
            shared_flags=defaults.shared_flags,
            shared_flag_mask=255,
            view=None,
        )
        common_filter.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                flags = np.array(ob.shared["flags"])
                flags |= ob.detdata["flags"][det]
                good = flags == 0
                check_rms = np.std(ob.detdata["signal"][det][good])
                # print(f"check_rms = {check_rms}, det rms = {rms[ob.name][det]}")
                self.assertLess(check_rms, 1e-3 * rms[ob.name][det])
        data.clear()
        del data
