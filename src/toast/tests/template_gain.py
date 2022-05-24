# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops
from ..templates import GainTemplate
from ..utils import rate_from_times
from ._helpers import create_outdir, create_satellite_data
from .mpi import MPITestCase


class TemplateGainTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_offset(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create a noise model from focalplane detector properties
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # make a simple pointing matrix
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=16,
            nest=False,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="I",
            detector_pointing=detpointing,
        )

        # Generate timestreams
        key = "signal"
        sim_dipole = ops.SimDipole(det_data=key, mode="solar", coord="G")
        sim_dipole.apply(data)
        key_template = "template"
        sim_dipole = ops.SimDipole(det_data="template", mode="solar", coord="G")
        sim_dipole.apply(data)

        # Set up binning operator for solving
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=noise_model.noise_model,
        )

        # set up a gain fluctuation template
        tmpl = GainTemplate(
            det_data=key,
            noise_model=noise_model.noise_model,
            template_name=key_template,
            order=1,
        )
        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        # Solve for template amplitudes
        calibration = ops.Calibrate(
            det_data="signal",
            result="calibrated",
            binning=binner,
            template_matrix=tmatrix,
        )
        calibration.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                np.testing.assert_allclose(
                    ob.detdata["calibrated"][det], np.ones(ob.n_local_samples)
                )

        del data
        return
