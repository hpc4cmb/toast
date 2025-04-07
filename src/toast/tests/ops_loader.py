# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
from datetime import datetime

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..data import Data
from ..mpi import MPI, Comm
from ..observation import default_values as defaults
from ..traits import Float, trait_docs
from .helpers import create_outdir, create_satellite_data
from .mpi import MPITestCase


class RandomLoader(object):
    """Class that generates random data."""

    def __init__(self, rms=1.0):
        self.rms = rms

    def load(self, obs):
        exists_data = obs.detdata.ensure(
            defaults.det_data,
            dtype=np.float64,
            detectors=obs.local_detectors,
            create_units=u.Kelvin,
        )
        for det in obs.local_detectors:
            obs.detdata[defaults.det_data][det] = np.random.normal(
                scale=self.rms, size=obs.n_local_samples
            )
        exists_flags = obs.detdata.ensure(
            defaults.det_flags, dtype=np.uint8, detectors=obs.local_detectors
        )

    def unload(self, obs):
        del obs.detdata[defaults.det_data]
        del obs.detdata[defaults.det_flags]


@trait_docs
class CheckRMS(ops.Operator):
    """Operator that just checks the data RMS"""

    expected = Float(
        1.0,
        help="Expected detector rms value",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        for ob in data.obs:
            for det in ob.local_detectors:
                rms = float(np.std(ob.detdata[defaults.det_data][det]))
                margin = np.sqrt(self.expected**4 / (ob.n_local_samples - 1))
                if np.absolute(rms - self.expected) > margin:
                    msg = (
                        f"{ob.name}[{det}]: {rms} outside {self.expected} +/- {margin}"
                    )
                    raise RuntimeError(msg)

    def _finalize(self, data, **kwargs):
        pass


class LoaderTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def create_data(self, rms=1.0):
        # Create a fake dataset, but delete any detector data
        data = create_satellite_data(self.comm, obs_per_group=2, flagged_pixels=False)
        ops.Delete(detdata=[defaults.det_data, defaults.det_flags]).apply(data)

        # Create a loader instance for each observation
        for obs in data.obs:
            obs.loader = RandomLoader(rms=rms)
        return data

    def test_load_exec(self):
        rms = 5.0
        data = self.create_data(rms=rms)

        # Verify that there is no data yet.
        for obs in data.obs:
            if defaults.det_data in obs.detdata:
                msg = f"Detector data in obs {obs.name} exists prior to load"
                print(msg)
                self.assertTrue(False)
            if defaults.det_flags in obs.detdata:
                msg = f"Detector flags in obs {obs.name} exists prior to load"
                print(msg)
                self.assertTrue(False)

        rms_checker = CheckRMS(expected=rms)
        rms_checker.load_exec(data)
        rms_checker.finalize(data)

        # Verify that the data was purged
        for obs in data.obs:
            if defaults.det_data in obs.detdata:
                msg = f"Detector data in obs {obs.name} was not unloaded"
                print(msg)
                self.assertTrue(False)
            if defaults.det_flags in obs.detdata:
                msg = f"Detector flags in obs {obs.name} was not unloaded"
                print(msg)
                self.assertTrue(False)
