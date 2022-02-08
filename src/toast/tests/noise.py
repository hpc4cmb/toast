# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import numpy as np

import h5py

from .mpi import MPITestCase

import numpy.testing as nt

from astropy import units as u

from ..noise import Noise

from ..noise_sim import AnalyticNoise

from ..io import H5File

from ._helpers import create_outdir


class InstrumentTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_noise_hdf5(self):
        detnames = ["det_01a", "det_01b", "det_02a", "det_02b"]
        streams = ["strm01", "strm02"]
        streams.extend(detnames)
        mix = {
            "det_01a": {"strm01": 0.25, "det_01a": 1.0},
            "det_01b": {"strm01": 0.75, "det_01b": 1.0},
            "det_02a": {"strm02": 0.25, "det_02a": 1.0},
            "det_02b": {"strm02": 0.75, "det_02b": 1.0},
        }
        freqs = np.linspace(0.0001, 5.0, num=50, endpoint=True)
        nse = Noise(
            detnames,
            {x: u.Quantity(freqs, u.Hz) for x in streams},
            {x: u.Quantity(np.ones(len(freqs)), u.K**2 * u.second) for x in streams},
            mixmatrix=mix,
        )

        nse_file = os.path.join(self.outdir, "noise.h5")

        with H5File(nse_file, "w", comm=self.comm) as f:
            nse.save_hdf5(f.handle, comm=self.comm)

        if self.comm is not None:
            self.comm.barrier()

        new_nse = Noise()
        with H5File(nse_file, "r", comm=self.comm) as f:
            new_nse.load_hdf5(f.handle, comm=self.comm)
        self.assertTrue(nse == new_nse)

    def test_analytic_hdf5(self):
        detnames = ["det_01a", "det_01b", "det_02a", "det_02b"]
        rate = {x: 10.0 * u.Hz for x in detnames}
        fmin = {x: 1e-5 * u.Hz for x in detnames}
        fknee = {x: 0.05 * u.Hz for x in detnames}
        alpha = {x: 1.0 for x in detnames}
        NET = {x: 1.0 * u.K * np.sqrt(1.0 * u.second) for x in detnames}

        nse = AnalyticNoise(
            detectors=detnames,
            rate=rate,
            fmin=fmin,
            fknee=fknee,
            alpha=alpha,
            NET=NET,
        )

        for droot in ["default", "serial"]:
            nse_file = os.path.join(self.outdir, f"sim_noise_{droot}.h5")
            with H5File(
                nse_file, "w", comm=self.comm, force_serial=(droot == "serial")
            ) as f:
                nse.save_hdf5(f.handle, comm=self.comm, force_serial=(droot == "serial"))

        if self.comm is not None:
            self.comm.barrier()

        for droot in ["default", "serial"]:
            nse_file = os.path.join(self.outdir, f"sim_noise_{droot}.h5")
            new_nse = AnalyticNoise()
            with H5File(nse_file, "r", comm=self.comm) as f:
                new_nse.load_hdf5(f.handle, comm=self.comm, force_serial=(droot == "serial"))
            self.assertTrue(nse == new_nse)
