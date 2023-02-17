# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import h5py
import numpy as np
import scipy.stats as stats
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from ..vis import set_matplotlib_backend
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class StatisticsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_statistics(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # This is a simulation with the same focalplane for every obs...
        sample_rate = data.obs[0].telescope.focalplane.sample_rate

        # Create a noise model from focalplane detector properties
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Simulate noise using this model
        sim_noise = ops.SimNoise()
        sim_noise.apply(data)

        # Measure TOD statistics
        statistics = ops.Statistics()
        statistics.output_dir = self.outdir
        statistics.apply(data)

        failed = False
        for obs in data.obs:
            fp = obs.telescope.focalplane
            if data.comm.group_rank != 0:
                continue
            if obs.name is not None:
                fname = os.path.join(self.outdir, f"{statistics.name}_{obs.name}.h5")
            else:
                fname = os.path.join(self.outdir, f"{statistics.name}_{obs.uid}.h5")
            # Load data
            with h5py.File(fname, "r") as f:
                detectors = list(f["detectors"].asstr())
                nsample = f.attrs["nsample"]
                ngood = f["ngood"][:].copy()
                mean = f["mean"][:].copy()
                var = f["variance"][:].copy()
                skew = f["skewness"][:].copy()
                kurt = f["kurtosis"][:].copy()
            # Test the statistics
            for det in obs.local_detectors:
                idet = detectors.index(det)
                sig = obs.detdata[defaults.det_data][det]
                # Test variance
                np.testing.assert_approx_equal(var[idet], np.var(sig), significant=6)
                # Test skewness
                np.testing.assert_approx_equal(
                    1 + skew[idet], 1 + stats.skew(sig), significant=6
                )
                # Test kurtosis
                np.testing.assert_approx_equal(
                    kurt[idet], stats.kurtosis(sig, fisher=False), significant=6
                )
        close_data(data)
