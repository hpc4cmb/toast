# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u
from astropy.table import Column, QTable

from ..instrument import Focalplane
from ..instrument_sim import fake_hexagon_focalplane
from ..io import H5File
from ._helpers import create_outdir
from .mpi import MPITestCase


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

        with H5File(fp_file, "w", comm=self.comm) as f:
            fp.save_hdf5(f.handle, comm=self.comm)

        if self.comm is not None:
            self.comm.barrier()

        newfp = Focalplane()

        with H5File(fp_file, "r", comm=self.comm) as f:
            newfp.load_hdf5(f.handle, comm=self.comm)

        self.assertTrue(newfp == fp)

    def test_focalplane_full(self):
        names = ["det_01a", "det_01b", "det_02a", "det_02b"]
        quats = [np.array([0, 0, 0, 1], dtype=np.float64) for x in range(len(names))]
        ndet = len(names)
        # Noise parameters (optional)
        psd_fmin = np.ones(ndet) * 1e-5 * u.Hz
        psd_fknee = np.ones(ndet) * 1e-2 * u.Hz
        psd_alpha = np.ones(ndet) * 1.0
        psd_NET = np.ones(ndet) * 1e-3 * u.K * u.s**0.5
        # Bandpass parameters (optional)
        bandcenter = np.ones(ndet) * 1e2 * u.GHz
        bandwidth = bandcenter * 0.1

        detdata = QTable(
            [
                names,
                quats,
                psd_fmin,
                psd_fknee,
                psd_alpha,
                psd_NET,
                bandcenter,
                bandwidth,
            ],
            names=[
                "name",
                "quat",
                "psd_fmin",
                "psd_fknee",
                "psd_alpha",
                "psd_net",
                "bandcenter",
                "bandwidth",
            ],
        )
        fp = Focalplane(detector_data=detdata, sample_rate=10.0 * u.Hz)

        fp_file = os.path.join(self.outdir, "focalplane_full.h5")
        check_file = os.path.join(self.outdir, "check_full.h5")

        with H5File(fp_file, "w", comm=self.comm) as f:
            fp.save_hdf5(f.handle, comm=self.comm)

        if self.comm is not None:
            self.comm.barrier()

        newfp = Focalplane()

        with H5File(fp_file, "r", comm=self.comm) as f:
            newfp.load_hdf5(f.handle, comm=self.comm)

        # Test getting noise PSD
        psd = newfp.noise.psd(names[-1])

        # Test convolving with bandpass
        freqs = np.linspace(50, 150, 100) * u.GHz
        values = np.linspace(0, 1, 100)
        result1 = newfp.bandpass.convolve(names[-1], freqs, values, rj=False)
        result2 = newfp.bandpass.convolve(names[-1], freqs, values, rj=True)

        with H5File(check_file, "w", comm=self.comm) as f:
            newfp.save_hdf5(f.handle, comm=self.comm)

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

        with H5File(fake_file, "w", comm=self.comm) as f:
            fp.save_hdf5(f.handle, comm=self.comm)

        if self.comm is not None:
            self.comm.barrier()
