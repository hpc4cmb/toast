# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np

from .. import ops as ops
from .. import qarray as qa
from .._libtoast import pixels_healpix, stokes_weights_IQU
from ..accelerator import ImplementationType, accel_enabled
from ..intervals import IntervalList, interval_dtype
from ..observation import default_values as defaults
from .helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class PointingHealpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_pointing_matrix_bounds(self):
        nside = 64
        npix = 12 * nside**2
        nest = True
        phivec = np.radians(
            [-360, -270, -180, -135, -90, -45, 0, 45, 90, 135, 180, 270, 360]
        )
        nsamp = phivec.size
        faketimes = -1.0 * np.ones(nsamp, dtype=np.float64)
        intervals = IntervalList(faketimes, samplespans=[(0, nsamp)])

        eps = np.array([0.0])
        gamma = np.array([0.0])
        cal = np.array([1.0])
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

        zero_index = np.array([0], dtype=np.int32)
        zero_flags = np.zeros(nsamp, dtype=np.uint8)

        n_pix_submap = npix
        hit_submaps = np.zeros(1, dtype=np.uint8)
        pixels_healpix(
            zero_index,
            quats.reshape(1, nsamp, 4),
            zero_flags,
            0,
            zero_index,
            pixels.reshape(1, nsamp),
            intervals.data,
            hit_submaps,
            n_pix_submap,
            nside,
            True,
            False,
        )
        stokes_weights_IQU(
            zero_index,
            quats.reshape(1, nsamp, 4),
            zero_index,
            weights.reshape(1, nsamp, nnz),
            hwpang,
            intervals.data,
            eps,
            gamma,
            cal,
            False,
            False,
        )
        failed = False
        bad = np.logical_or(pixels < 0, pixels > npix - 1)
        nbad = np.sum(bad)
        if nbad > 0:
            print(f"{nbad} pixels are outside of the map.")
            print(f"phi = {np.degrees(phivec[bad])} deg, pix = {pixels[bad]}")
            failed = True
        self.assertFalse(failed)
        return

    def test_pointing_matrix_weights(self):
        nside = 64
        nest = True
        psivec = np.radians([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        expected_Q = np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0])
        expected_U = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0])
        nsamp = psivec.size

        eps = np.array([0.0])
        gamma = np.array([0.0])
        cal = np.array([1.0])
        mode = "IQU"
        nnz = 3

        # With no HWP, we set this to some array whose length is not
        # nsamp.
        hwpang = np.zeros(1, dtype=np.float64)

        flags = np.zeros(nsamp, dtype=np.uint8)
        weights = np.zeros([nsamp, nnz], dtype=np.float64)
        zero_index = np.array([0], dtype=np.int32)
        zero_flags = np.zeros(nsamp, dtype=np.uint8)
        faketimes = -1.0 * np.ones(nsamp, dtype=np.float64)
        intervals = IntervalList(faketimes, samplespans=[(0, nsamp)])

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
        stokes_weights_IQU(
            zero_index,
            quats.reshape(1, nsamp, 4),
            zero_index,
            weights.reshape(1, nsamp, nnz),
            hwpang,
            intervals.data,
            eps,
            gamma,
            cal,
            False,
            False,
        )
        weights_ref = []
        for q, u in zip(expected_Q, expected_U):
            weights_ref.append(np.array([1, q, u]))
        weights_ref = np.vstack(weights_ref)
        failed = False
        for w1, w2 in zip(weights_ref, weights):
            if not np.allclose(w1, w2):
                print(
                    f"Pointing weights do not agree: {w2} != {w1}",
                    flush=True,
                )
                failed = True
            else:
                print("Pointing weights agree: {} == {}".format(w1, w2), flush=True)
                pass
        self.assertFalse(failed)
        return

    def test_pointing_matrix_weights_hwp(self):
        nside = 64
        nest = True
        psivec = np.radians([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        expected_Q = np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0])
        expected_U = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0])
        nsamp = len(psivec)

        eps = np.array([0.0])
        gamma = np.array([0.0])
        cal = np.array([1.0])
        mode = "IQU"
        nnz = 3

        flags = np.zeros(nsamp, dtype=np.uint8)
        zero_index = np.array([0], dtype=np.int32)
        zero_flags = np.zeros(nsamp, dtype=np.uint8)
        faketimes = -1.0 * np.ones(nsamp, dtype=np.float64)
        intervals = IntervalList(faketimes, samplespans=[(0, nsamp)])

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

        # With HWP angle == 0.0
        hwpang = np.zeros(nsamp)
        weights_zero = np.zeros([nsamp, nnz], dtype=np.float64)

        stokes_weights_IQU(
            zero_index,
            quats.reshape(1, nsamp, 4),
            zero_index,
            weights_zero.reshape(1, nsamp, nnz),
            hwpang,
            intervals.data,
            eps,
            gamma,
            cal,
            False,
            False,
        )

        failed = False
        if not np.allclose(weights_zero[:, 1], expected_Q):
            msg = f"Q weights_zero do not match expected values {weights_zero[:, 1]}"
            msg += f" != {expected_Q}"
            print(msg)
            failed = True
        if not np.allclose(weights_zero[:, 2], expected_U):
            msg = f"U weights_zero do not match expected values {weights_zero[:, 1]}"
            msg += f" != {expected_U}"
            print(msg)
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
                [(times[0], times[-1], 0, nsample)], dtype=interval_dtype
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
