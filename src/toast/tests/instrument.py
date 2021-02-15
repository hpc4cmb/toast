# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .mpi import MPITestCase

import numpy.testing as nt

from astropy import units as u

from astropy.table import QTable, Column

from ..instrument import Focalplane

from ..instrument_sim import fake_hexagon_focalplane

from ._helpers import create_outdir


class InstrumentTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_focalplane(self):
        names = ["det_01a", "det_01b", "det_02a", "det_02b"]
        quats = [np.array([0, 0, 0, 1], dtype=np.float64) for x in range(len(names))]
        detdata = QTable([names, quats], names=["name", "quat"])
        fp = Focalplane(detector_data=detdata, sample_rate=10.0 * u.Hz)

        fp_file = os.path.join(self.outdir, "focalplane.h5")
        check_file = os.path.join(self.outdir, "check.h5")

        if self.comm is None or self.comm.rank == 0:
            fp.write(fp_file)
        if self.comm is not None:
            self.comm.barrier()

        newfp = Focalplane(file=fp_file)

        if self.comm is None or self.comm.rank == 0:
            newfp.write(check_file)
        if self.comm is not None:
            self.comm.barrier()

    def test_sim_focalplane(self):
        fp = fake_hexagon_focalplane(
            n_pix=7,
            width=5.0 * u.degree,
            sample_rate=100.0 * u.Hz,
            epsilon=0.05,
            fwhm=10.0 * u.arcmin,
            bandcenter=150 * u.Hz,
            bandwidth=20 * u.Hz,
            psd_net=0.05 * u.K * np.sqrt(1 * u.second),
            psd_fmin=1.0e-5 * u.Hz,
            psd_alpha=1.2,
            psd_fknee=0.05 * u.Hz,
        )
        fake_file = os.path.join(self.outdir, "fake_hex.h5")

        if self.comm is None or self.comm.rank == 0:
            fp.write(fake_file)
        if self.comm is not None:
            self.comm.barrier()
