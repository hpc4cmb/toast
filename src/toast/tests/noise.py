# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import h5py
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..instrument_sim import fake_hexagon_focalplane
from ..io import H5File
from ..noise import Noise
from ..noise_sim import AnalyticNoise
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase
from .ops_noise_estim import plot_noise_estim_compare


class InstrumentTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_noise_hdf5(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        for ob in data.obs:
            detnames = ob.local_detectors
            streams = ["strm01", "strm02"]
            streams.extend(detnames)
            mix = dict()
            for idet, det in enumerate(detnames):
                mix[det] = {
                    "strm01": 0.25 + 0.5 * (idet % 2),
                    "strm02": 0.25 + 0.5 * ((idet + 1) % 2),
                }
            freqs = np.linspace(0.0001, 5.0, num=50, endpoint=True)
            nse = Noise(
                detnames,
                {x: u.Quantity(freqs, u.Hz) for x in streams},
                {
                    x: u.Quantity(np.ones(len(freqs)), u.K**2 * u.second)
                    for x in streams
                },
                mixmatrix=mix,
            )

            nse_file = os.path.join(self.outdir, f"{ob.name}_noise.h5")

            with H5File(nse_file, "w", comm=ob.comm.comm_group) as f:
                nse.save_hdf5(f.handle, ob)

            if ob.comm.comm_group is not None:
                ob.comm.comm_group.barrier()

            new_nse = Noise()
            with H5File(nse_file, "r", comm=ob.comm.comm_group) as f:
                new_nse.load_hdf5(f.handle, ob)
            self.assertTrue(nse == new_nse)

        close_data(data)

    def test_analytic_hdf5(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        for ob in data.obs:
            detnames = ob.local_detectors

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

            nse_file = os.path.join(self.outdir, f"{ob.name}_sim_noise.h5")
            with H5File(nse_file, "w", comm=ob.comm.comm_group) as f:
                nse.save_hdf5(f.handle, ob)

            if ob.comm.comm_group is not None:
                ob.comm.comm_group.barrier()

            new_nse = AnalyticNoise()
            with H5File(nse_file, "r", comm=ob.comm.comm_group) as f:
                new_nse.load_hdf5(f.handle, ob)
            self.assertTrue(nse == new_nse)
        close_data(data)

    def test_noise_fit(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create a noise model from focalplane detector properties
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Fit the noise model
        noise_fitter = ops.FitNoiseModel(
            noise_model=noise_model.noise_model, out_model="fit_noise"
        )
        noise_fitter.apply(data)

        for ob in data.obs:
            in_model = ob[noise_model.noise_model]
            out_model = ob[noise_fitter.out_model]
            for det in ob.local_detectors:
                in_psd = in_model.psd(det)
                fit_psd = out_model.psd(det)

                fname = os.path.join(self.outdir, f"fit_{ob.name}_{det}.png")
                plot_noise_estim_compare(
                    fname,
                    in_model.NET(det),
                    in_model.freq(det),
                    in_model.psd(det),
                    in_model.freq(det),
                    in_model.psd(det),
                    fit_freq=out_model.freq(det),
                    fit_psd=out_model.psd(det),
                )

                np.testing.assert_array_almost_equal(
                    in_psd.value, fit_psd.value, decimal=2
                )

        close_data(data)
