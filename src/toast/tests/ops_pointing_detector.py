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
from ..vis import plot_projected_quats
from .helpers import (
    close_data,
    create_ground_data,
    create_outdir,
    create_satellite_data,
)
from .mpi import MPITestCase


class PointingDetectorTest(MPITestCase):
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
            handle = open(
                os.path.join(self.outdir, "out_test_detpointing_simple_info"), "w"
            )
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
                self.assertTrue(rms_theta > 0.5 and rms_theta < 1.5)
                self.assertTrue(rms_phi > 0.5 and rms_phi < 1.5)
        close_data(data)

    def test_detector_pointing_hwp_deflect_plot(self):
        # Create fake observing.
        data = create_ground_data(self.comm, sample_rate=30.0 * u.Hz, hwp_rpm=60.0)

        # Regular pointing in Az/El
        detpointing1 = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel,
        )
        detpointing1.apply(data)

        # Pointing with deflection in Az/El
        detpointing2 = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel,
            hwp_angle=defaults.hwp_angle,
            hwp_angle_offset=0 * u.deg,
            # hwp_angle_offset=45 * u.deg,
            hwp_deflection_radius=2.0 * u.deg,
            quats="deflected",
        )
        detpointing2.apply(data)

        # Plot pointing.  We take the first detector (which will be at the
        # boresight) for the nominal and deflected cases and verify that
        # the motion makes sense.
        if data.comm.world_rank == 0 and self.make_plots:
            n_debug = 30
            n_skip = 1
            start = 150
            slc = slice(start, start + n_debug * n_skip, n_skip)
            bquat = np.array(data.obs[0].shared[defaults.boresight_azel].data[slc, :])
            dquat = np.zeros((2, n_debug, 4), dtype=np.float64)
            dquat[0] = data.obs[0].detdata[detpointing1.quats][0, slc, :]
            dquat[1] = data.obs[0].detdata[detpointing2.quats][0, slc, :]
            invalid = np.array(data.obs[0].shared[defaults.shared_flags][slc])
            invalid &= defaults.shared_mask_invalid
            valid = np.logical_not(invalid)
            outfile = os.path.join(self.outdir, "pointing_deflection.pdf")
            plot_projected_quats(
                outfile,
                qbore=bquat,
                qdet=dquat,
                valid=valid,
                scale=2.0,
                equal_aspect=False,
            )
        close_data(data)
