# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops
from ..accelerator import ImplementationType, accel_data_table, accel_enabled
from ..observation import default_values as defaults
from ..templates import AmplitudesMap, Offset
from ..utils import rate_from_times
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


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

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = (
            data.obs[0].shared[defaults.times][-1]
            - data.obs[0].shared[defaults.times][0]
        )
        step_seconds = float(int(ob_time / 10.0))

        tmpl = Offset(
            det_data=defaults.det_data,
            det_flags=None,
            times=defaults.times,
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
                np.testing.assert_equal(ob.detdata[defaults.det_data][det], 1.0)

        # Accumulate amplitudes
        for det in tmpl.detectors():
            for ob in data.obs:
                tmpl.project_signal(det, amps)

        # Verify
        for ob in data.obs:
            # Get the step boundaries
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
                ob.shared[defaults.times]
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
                        np.sum(ob.detdata[defaults.det_data][det, slc]), 1.0 * sz
                    )

        close_data(data)

    def test_accel(self):
        if not accel_enabled():
            print("Accelerator use not enabled, skipping test")
            return

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create a default noise model
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = (
            data.obs[0].shared[defaults.times][-1]
            - data.obs[0].shared[defaults.times][0]
        )
        step_seconds = float(int(ob_time / 10.0))

        tmpl = Offset(
            det_data=defaults.det_data,
            det_flags=None,
            times=defaults.times,
            noise_model=noise_model.noise_model,
            step_time=step_seconds * u.second,
        )
        # Set the data
        tmpl.data = data

        # Get some amplitudes and set to one
        amps = tmpl.zeros()
        amps.local[:] = 1.0

        data_names = {
            "detdata": [defaults.det_data],
            "shared": [],
            "global": [],
            "meta": [],
            "intervals": [None],
        }

        data.accel_create(data_names)
        data.accel_update_device(data_names)
        amps.accel_create("Offset")
        amps.accel_update_device()

        # Project.
        for det in tmpl.detectors():
            for ob in data.obs:
                tmpl.add_to_signal(det, amps, use_accel=True)

        # Accumulate amplitudes
        for det in tmpl.detectors():
            for ob in data.obs:
                tmpl.project_signal(det, amps, use_accel=True)

        data.accel_update_host(data_names)
        amps.accel_update_host()

        # Verify
        for ob in data.obs:
            # Get the step boundaries
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
                ob.shared[defaults.times]
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
                        np.sum(ob.detdata[defaults.det_data][det, slc]), 1.0 * sz
                    )

        close_data(data)

    def test_accel_pipeline(self):
        if not accel_enabled():
            print("Accelerator use not enabled, skipping test")
            return

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create a default noise model
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = (
            data.obs[0].shared[defaults.times][-1]
            - data.obs[0].shared[defaults.times][0]
        )
        step_seconds = float(int(ob_time / 10.0))

        tmpl = Offset(
            name="offset",
            times=defaults.times,
            noise_model=noise_model.noise_model,
            step_time=step_seconds * u.second,
        )

        # Create the template matrix
        amp_key = "amps"
        tmatrix = ops.TemplateMatrix(
            det_data=defaults.det_data,
            det_flags=None,
            amplitudes=amp_key,
            transpose=True,
            templates=[tmpl],
        )

        # Create some zero amplitudes
        for ob in data.obs:
            ob.detdata[defaults.det_data].update_units(1.0 / defaults.det_data_units)
        tmatrix.apply(data)

        # Set the amplitudes to one
        data[amp_key]["offset"].local[:] = 1.0

        # Create a single-detector temporary timestream to test the re-use
        # of memory with the change_detectors() method
        det_temp = "temp"

        forward_tmatrix = ops.TemplateMatrix(
            name="offset",
            det_data=det_temp,
            det_data_units=1.0 / defaults.det_data_units,
            det_flags=None,
            amplitudes=amp_key,
            templates=[tmpl],
            transpose=False,
        )
        backward_tmatrix = forward_tmatrix.duplicate()
        backward_tmatrix.det_data_units = defaults.det_data_units
        backward_tmatrix.transpose = True

        # Use a pipeline to execute on accelerator
        pipe = ops.Pipeline(
            detector_sets=["SINGLE"],
            operators=[forward_tmatrix, backward_tmatrix],
        )

        # First call initializes templates, so we must disable accel use
        pipe.apply(data, use_accel=False)

        # Reset data and now run on accel
        for ob in data.obs:
            ob.detdata[det_temp].clear()
            del ob.detdata[det_temp]
            data[amp_key]["offset"].local[:] = 1.0

        # data.info()
        # accel_data_table()

        pipe.apply(data)

        # Construct the expected amplitude values.  All observations have
        # the same detector list in this case.  We are accumulating to the
        # input amplitudes (with starting value 1.0) without clearing those,
        # so we add 1.0 to the expected values.
        offset = 0
        all_dets = list(data.obs[0].local_detectors)
        expected = list()
        for det in all_dets:
            for iob, ob in enumerate(data.obs):
                # Get the step boundaries
                (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
                    ob.shared[defaults.times]
                )
                step_samples = int(step_seconds * rate + 0.5)

                n_step = ob.n_local_samples // step_samples
                if n_step * step_samples < ob.n_local_samples:
                    n_step += 1
                sizes = [step_samples for x in range(n_step - 1)]
                sizes.append(ob.n_local_samples - (n_step - 1) * step_samples)
                expected.extend(sizes)
                offset += len(sizes)

        expected = 1.0 + np.array(expected, dtype=np.float64)

        np.testing.assert_almost_equal(
            data[amp_key]["offset"].local,
            expected,
        )

        close_data(data)

    def test_compare(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)
        for ob in data.obs:
            ob.detdata.create("pydata")

        # Create a default noise model
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = (
            data.obs[0].shared[defaults.times][-1]
            - data.obs[0].shared[defaults.times][0]
        )
        step_seconds = float(int(ob_time / 10.0))

        tmpl = Offset(
            det_data=defaults.det_data,
            times=defaults.times,
            noise_model=noise_model.noise_model,
            step_time=step_seconds * u.second,
        )
        pytmpl = Offset(
            det_data="pydata",
            times=defaults.times,
            noise_model=noise_model.noise_model,
            step_time=step_seconds * u.second,
            kernel_implementation=ImplementationType.NUMPY,
        )

        # Set the data
        tmpl.data = data
        pytmpl.data = data

        # Get some amplitudes and set to one
        amps = tmpl.zeros()
        amps.local[:] = 1.0
        pyamps = pytmpl.zeros()
        pyamps.local[:] = 1.0

        # Project.
        for det in tmpl.detectors():
            for ob in data.obs:
                tmpl.add_to_signal(det, amps)
        for det in pytmpl.detectors():
            for ob in data.obs:
                pytmpl.add_to_signal(det, pyamps)

        for ob in data.obs:
            np.testing.assert_allclose(
                ob.detdata[defaults.det_data], ob.detdata["pydata"]
            )

        # Accumulate amplitudes
        for det in tmpl.detectors():
            for ob in data.obs:
                tmpl.project_signal(det, amps)
        for det in pytmpl.detectors():
            for ob in data.obs:
                pytmpl.project_signal(det, pyamps)

        # Verify
        np.testing.assert_allclose(amps.local, pyamps.local)

        close_data(data)
