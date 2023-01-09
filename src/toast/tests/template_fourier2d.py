# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops
from ..observation import default_values as defaults
from ..templates import Fourier2D
from ..utils import rate_from_times
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class TemplateFourier2DTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_projection(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create a default noise model
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        tmpl = Fourier2D(
            det_data=defaults.det_data,
            times=defaults.times,
            noise_model=noise_model.noise_model,
        )

        # Set the data
        tmpl.data = data

        # Get some amplitudes and set to one
        amps = tmpl.zeros()
        amps.local[:] = 1.0

        # Project.
        for det in tmpl.detectors():
            for ob in data.obs:
                tmpl.add_to_signal(det, amps)

        # Verify
        if self.comm is None or self.comm.rank == 0:
            print("\n\nNOTE:  Fourier2D template unit tests incomplete\n", flush=True)
        # for ob in data.obs:
        #     for det in ob.local_detectors:
        #         np.testing.assert_equal(ob.detdata["signal"][det], 1.0)

        # Accumulate amplitudes
        for det in tmpl.detectors():
            for ob in data.obs:
                tmpl.project_signal(det, amps)

        # Verify
        # FIXME...

        close_data(data)
