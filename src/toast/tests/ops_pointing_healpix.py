# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np

from .. import ops as ops
from .. import qarray as qa
from .._libtoast import healpix_pixels, stokes_weights
from ..accelerator import ImplementationType, accel_enabled
from ..healpix import HealpixPixels
from ..intervals import IntervalList, interval_dtype
from ..observation import default_values as defaults
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class PointingHealpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_pointing_matrix_healpix2(self):
        nside = 64
        npix = 12 * nside**2
        hpix = HealpixPixels(64)
        nest = True
        phivec = np.radians(
            [-360, -270, -180, -135, -90, -45, 0, 45, 90, 135, 180, 270, 360]
        )
        nsamp = phivec.size
        eps = 0.0
        cal = 1.0
        mode = "IQU"
        nnz = 3
        hwpang = np.zeros(nsamp)
        flags = np.zeros(nsamp, dtype=np.uint8)
        pixels = np.zeros(nsamp, dtype=np.int64)
        weights = np.zeros([nsamp, nnz], dtype=np.float64)
        theta = np.radians(135)
        psi = np.radians(135)
        quats = []
        xaxis, yaxis, zaxis = np.eye(3)
        for phi in phivec:
            phirot = qa.rotation(zaxis, phi)
            quats.append(qa.from_iso_angles(theta, phi, psi))
        quats = np.vstack(quats)
        healpix_pixels(
            hpix,
            nest,
            quats.reshape(-1),
            flags,
            pixels,
        )
        stokes_weights(
            eps,
            cal,
            mode,
            quats.reshape(-1),
            hwpang,
            flags,
            weights.reshape(-1),
        )
        failed = False
        bad = np.logical_or(pixels < 0, pixels > npix - 1)
        nbad = np.sum(bad)
        if nbad > 0:
            print(
                "{} pixels are outside of the map. phi = {} deg".format(
                    nbad, np.degrees(phivec[bad])
                )
            )
            failed = True
        self.assertFalse(failed)
        return

    def test_pointing_matrix_healpix(self):
        nside = 64
        hpix = HealpixPixels(64)
        nest = True
        psivec = np.radians([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        # psivec = np.radians([-180, 180])
        nsamp = psivec.size
        eps = 0.0
        cal = 1.0
        mode = "IQU"
        nnz = 3
        hwpang = np.zeros(nsamp)
        flags = np.zeros(nsamp, dtype=np.uint8)
        weights = np.zeros([nsamp, nnz], dtype=np.float64)
        pix = 49103
        theta, phi = hp.pix2ang(nside, pix, nest=nest)
        xaxis, yaxis, zaxis = np.eye(3)
        thetarot = qa.rotation(yaxis, theta)
        phirot = qa.rotation(zaxis, phi)
        pixrot = qa.mult(phirot, thetarot)
        quats = []
        for psi in psivec:
            psirot = qa.rotation(zaxis, psi)
            quats.append(qa.mult(pixrot, psirot))
        quats = np.vstack(quats)
        stokes_weights(
            eps,
            cal,
            mode,
            quats.reshape(-1),
            hwpang,
            flags,
            weights.reshape(-1),
        )
        weights_ref = []
        for quat in quats:
            theta, phi, psi = qa.to_iso_angles(quat)
            weights_ref.append(np.array([1, np.cos(2 * psi), np.sin(2 * psi)]))
        weights_ref = np.vstack(weights_ref)
        failed = False
        for w1, w2, psi, quat in zip(weights_ref, weights, psivec, quats):
            # print("\npsi = {}, quat = {} : ".format(psi, quat), end="")
            if not np.allclose(w1, w2):
                print(
                    "Pointing weights do not agree: {} != {}".format(w1, w2), flush=True
                )
                failed = True
            else:
                # print("Pointing weights agree: {} == {}".format(w1, w2), flush=True)
                pass
        self.assertFalse(failed)
        return

    def test_pointing_matrix_healpix_hwp(self):
        nside = 64
        hpix = HealpixPixels(64)
        nest = True
        psivec = np.radians([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        nsamp = len(psivec)
        eps = 0.0
        cal = 1.0
        mode = "IQU"
        nnz = 3
        flags = np.zeros(nsamp, dtype=np.uint8)
        pix = 49103
        theta, phi = hp.pix2ang(nside, pix, nest=nest)
        xaxis, yaxis, zaxis = np.eye(3)
        thetarot = qa.rotation(yaxis, theta)
        phirot = qa.rotation(zaxis, phi)
        pixrot = qa.mult(phirot, thetarot)
        quats = []
        for psi in psivec:
            psirot = qa.rotation(zaxis, psi)
            quats.append(qa.mult(pixrot, psirot))
        quats = np.vstack(quats)

        # First with HWP angle == 0.0
        hwpang = np.zeros(nsamp)
        weights_zero = np.zeros([nsamp, nnz], dtype=np.float64)
        stokes_weights(
            eps,
            cal,
            mode,
            quats.reshape(-1),
            hwpang,
            flags,
            weights_zero.reshape(-1),
        )

        # Now passing hwpang == None
        weights_none = np.zeros([nsamp, nnz], dtype=np.float64)
        stokes_weights(
            eps,
            cal,
            mode,
            quats.reshape(-1),
            None,
            flags,
            weights_none.reshape(-1),
        )
        # print("")
        # for i in range(nsamp):
        #     print(
        #         "HWP zero:  {} {} | {} {} {}".format(
        #             psivec[i],
        #             pixels_zero[i],
        #             weights_zero[i][0],
        #             weights_zero[i][1],
        #             weights_zero[i][2],
        #         )
        #     )
        #     print(
        #         "    none:  {} {} | {} {} {}".format(
        #             psivec[i],
        #             pixels_none[i],
        #             weights_none[i][0],
        #             weights_none[i][1],
        #             weights_none[i][2],
        #         )
        #     )
        failed = False

        if not np.allclose(weights_zero, weights_none):
            print(
                "HWP weights do not agree {} != {}".format(weights_zero, weights_none)
            )
            failed = True

        self.assertFalse(failed)
        return

    def test_hpix_simple(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            detector_pointing=detpointing,
        )
        pixels.apply(data)

        # Also make a copy using a python codepath
        detpointing.kernel_implementation = ImplementationType.NUMPY
        detpointing.quats = "pyquat"
        pixels.kernel_implementation = ImplementationType.NUMPY
        pixels.quats = "pyquat"
        pixels.pixels = "pypix"
        pixels.apply(data)

        for ob in data.obs:
            np.testing.assert_array_equal(
                ob.detdata[defaults.pixels], ob.detdata["pypix"]
            )

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        handle = None
        if rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_hpix_simple_info"), "w")
        data.info(handle=handle)
        if rank == 0:
            handle.close()

        close_data(data)

    def test_pixweight_pipe(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="I",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        pipe = ops.Pipeline(operators=[pixels, weights])
        pipe.apply(data)

        # Also make a copy using a python codepath
        detpointing.kernel_implementation = ImplementationType.NUMPY
        detpointing.quats = "pyquat"
        pixels.kernel_implementation = ImplementationType.NUMPY
        pixels.quats = "pyquat"
        pixels.pixels = "pypixels"
        pixels.apply(data)
        weights.kernel_implementation = ImplementationType.NUMPY
        weights.quats = "pyquat"
        weights.weights = "pyweight"
        weights.apply(data)

        for ob in data.obs:
            np.testing.assert_allclose(
                ob.detdata[defaults.weights], ob.detdata["pyweight"]
            )
            np.testing.assert_equal(ob.detdata[defaults.pixels], ob.detdata["pypixels"])

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        handle = None
        if rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_pixweight_pipe"), "w")
        data.info(handle=handle)
        if rank == 0:
            handle.close()

        close_data(data)

    def test_pixweight_accel(self):
        if not accel_enabled():
            print("Accelerator not enabled, skipping test")
            return

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="I",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )

        pipe = ops.Pipeline(operators=[pixels, weights])

        # Compute
        pipe.apply(data, use_accel=None)

        # Also make a copy using a python codepath
        detpointing.kernel_implementation = ImplementationType.NUMPY
        detpointing.quats = "pyquat"
        pixels.kernel_implementation = ImplementationType.NUMPY
        pixels.quats = "pyquat"
        pixels.pixels = "pypixels"
        pixels.apply(data)
        weights.kernel_implementation = ImplementationType.NUMPY
        weights.quats = "pyquat"
        weights.weights = "pyweight"
        weights.apply(data)

        for ob in data.obs:
            np.testing.assert_allclose(
                ob.detdata[defaults.weights], ob.detdata["pyweight"]
            )
            np.testing.assert_equal(ob.detdata[defaults.pixels], ob.detdata["pypixels"])
        close_data(data)

    def test_hpix_interval(self):
        data = create_satellite_data(self.comm)

        full_intervals = "full_intervals"
        half_intervals = "half_intervals"
        for obs in data.obs:
            times = obs.shared[defaults.times]
            nsample = len(times)
            intervals1 = np.array(
                [(times[0], times[-1], 0, nsample - 1)], dtype=interval_dtype
            ).view(np.recarray)
            intervals2 = np.array(
                [(times[0], times[nsample // 2], 0, nsample // 2)], dtype=interval_dtype
            ).view(np.recarray)
            obs.intervals[full_intervals] = IntervalList(times, intervals=intervals1)
            obs.intervals[half_intervals] = IntervalList(times, intervals=intervals2)

        detpointing = ops.PointingDetectorSimple(view=half_intervals)
        pixels = ops.PixelsHealpix(
            nside=64,
            detector_pointing=detpointing,
            view=full_intervals,
        )
        with self.assertRaises(RuntimeError):
            pixels.apply(data)

        # NOTE:  the detector pointing operator has no way of knowing that it is being
        # called again with a different set of intervals.  It will see the existing
        # detector pointing object and skip it.  So we delete it first.
        ops.Delete(detdata=[detpointing.quats, pixels.pixels]).apply(data)

        detpointing.view = full_intervals
        pixels.view = half_intervals
        pixels.apply(data)

        close_data(data)

    def test_weights_hwpnull(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        detpointing = ops.PointingDetectorSimple()
        weights = ops.StokesWeights(mode="IQU", detector_pointing=detpointing)
        weights.apply(data)

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        handle = None
        if rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_weights_hwpnull"), "w")
        data.info(handle=handle)
        if rank == 0:
            handle.close()

        close_data(data)
