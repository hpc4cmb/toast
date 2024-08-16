# Copyright (c) 2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import astropy.units as u
import healpy as hp
import numpy as np

from .. import ops as ops
from .. import qarray as qa
from .._libtoast import pixels_healpix, stokes_weights_IQU
from ..accelerator import ImplementationType, accel_enabled
from ..intervals import IntervalList, interval_dtype
from ..observation import default_values as defaults
from ._helpers import (
    close_data, create_outdir, create_satellite_data, create_ground_data
)
from .mpi import MPITestCase


class PointingDetectorTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_detector_pointing_simple(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        detpointing = ops.PointingDetectorSimple()
        detpointing.apply(data)

        # Also make a copy using a python codepath
        detpointing.kernel_implementation = ImplementationType.NUMPY
        detpointing.quats = "pyquat"
        detpointing.apply(data)

        for ob in data.obs:
            np.testing.assert_array_equal(
                ob.detdata[defaults.quats], ob.detdata["pyquat"]
            )

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        handle = None
        if rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_detpointing_simple_info"), "w")
        data.info(handle=handle)
        if rank == 0:
            handle.close()

        close_data(data)

    def test_detector_pointing_hwp_deflect(self):
        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Regular pointing
        detpointing1 = ops.PointingDetectorSimple()
        detpointing1.apply(data)

        # Pointing with deflection
        detpointing2 = ops.PointingDetectorSimple(
            hwp_angle=defaults.hwp_angle,
            hwp_angle_offset=15 * u.deg,
            hwp_deflection_radius=1.0 * u.deg,
            quats="deflected",
        )
        detpointing2.apply(data)

        # Compare
        for ob in data.obs:
            quats1 = ob.detdata[defaults.quats]
            quats2 = ob.detdata["deflected"]
            ndet = quats1.shape[0]
            for idet in range(ndet):
                theta1, phi1, psi1 = qa.to_iso_angles(quats1[idet])
                theta2, phi2, psi2 = qa.to_iso_angles(quats2[idet])
                rms_theta = np.degrees(np.std(theta1 - theta2))
                rms_phi = np.degrees(np.std(phi1 - phi2))
                self.assertTrue(rms_theta > .5 and rms_theta < 1.5)
                self.assertTrue(rms_phi > .5 and rms_phi < 1.5)

        close_data(data)
