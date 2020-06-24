# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

import numpy as np
import numpy.testing as nt

from ..mpi import Comm, MPI

from ..data import Data

from ..instrument import Focalplane, Telescope

from ..instrument_sim import fake_hexagon_focalplane

from .. import future_ops as ops

from ._helpers import create_outdir, create_comm


class SimSatelliteTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.toastcomm = create_comm(self.comm)

        npix = 1
        ring = 1
        while 2 * npix < self.toastcomm.group_size:
            npix += 6 * ring
            ring += 1
        self.fp = fake_hexagon_focalplane(n_pix=npix)
        self.tele = Telescope("test", focalplane=self.fp)
        self.simsat = ops.SimSatellite(n_observation=2, telescope=self.tele)

    def test_exec(self):
        data = Data(self.toastcomm)
        self.simsat.exec(data)
