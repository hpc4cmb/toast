# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from astropy import units as u

from .mpi import MPITestCase

from ..utils import rate_from_times

from .. import ops

from ..observation import default_names as obs_names

from ..templates import Offset

from ._helpers import create_outdir, create_satellite_data


class TemplateOffsetTest(MPITestCase):
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

        # Create some empty detector data
        for ob in data.obs:
            ob.detdata.create(obs_names.det_data, dtype=np.float64)

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = (
            data.obs[0].shared[obs_names.times][-1]
            - data.obs[0].shared[obs_names.times][0]
        )
        step_seconds = float(int(ob_time / 10.0))

        tmpl = Offset(
            det_data=obs_names.det_data,
            times=obs_names.times,
            noise_model=noise_model.noise_model,
            step_time=step_seconds * u.second,
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
        for ob in data.obs:
            for det in ob.local_detectors:
                np.testing.assert_equal(ob.detdata[obs_names.det_data][det], 1.0)

        # Accumulate amplitudes
        for det in tmpl.detectors():
            for ob in data.obs:
                tmpl.project_signal(det, amps)

        # Verify
        for ob in data.obs:
            # Get the step boundaries
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
                ob.shared[obs_names.times]
            )
            step_samples = int(step_seconds * rate)
            n_step = ob.n_local_samples // step_samples
            slices = [
                slice(x * step_samples, (x + 1) * step_samples, 1)
                for x in range(n_step - 1)
            ]
            sizes = [step_samples for x in range(n_step - 1)]
            slices.append(slice((n_step - 1) * step_samples, ob.n_local_samples, 1))
            sizes.append(ob.n_local_samples - (n_step - 1) * step_samples)

            for det in ob.local_detectors:
                for slc, sz in zip(slices, sizes):
                    np.testing.assert_equal(
                        np.sum(ob.detdata[obs_names.det_data][det, slc]), 1.0 * sz
                    )

        del data
        return
