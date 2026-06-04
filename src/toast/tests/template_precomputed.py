# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops
from ..observation import default_values as defaults
from ..templates import PreComputed
from ..vis import plot_healpix_maps, plot_noise_estim
from .helpers import (
    close_data,
    create_fake_healpix_scanned_tod,
    create_fake_wcs_scanned_tod,
    create_ground_data,
    create_outdir,
    create_satellite_data,
)
from .mpi import MPITestCase


class TemplatePrecomputedTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        self.nside = 64
        np.random.seed(123456)
        if (
            ("CONDA_BUILD" in os.environ)
            or ("CIBUILDWHEEL" in os.environ)
            or ("CI" in os.environ)
        ):
            self.make_plots = False
        else:
            self.make_plots = True

    def fake_templates(self, data, obs_key):
        # Create fake templates and coadd to detectors with random
        # coefficients.
        for ob in data.obs:
            nsamp = ob.n_local_samples
            amp_ptp = 0.0
            coeff = np.random.uniform(low=0.5, high=1.5, size=len(ob.local_detectors))
            for det in ob.local_detectors:
                dptp = np.ptp(ob.detdata[defaults.det_data][det])
                amp_ptp = max(amp_ptp, dptp)
            tnames = list()
            index = np.arange(nsamp, dtype=np.int64)
            templates = dict()
            for it in range(5):
                name = f"template_{it:05d}"
                tnames.append(name)
                templates[name] = amp_ptp * np.sin(np.pi * (it + 5) * index / nsamp)
            det_to_key = dict()
            det_coeff = dict()
            for idet, det in enumerate(ob.local_detectors):
                it = idet % 5
                name = f"template_{it:05d}"
                det_to_key[det] = name
                det_coeff[det] = coeff[idet]
                ob.detdata[defaults.det_data][det] += coeff[idet] * templates[name]
            templates["det_to_key"] = det_to_key
            ob[obs_key] = templates
            ob["template_amp"] = amp_ptp
            ob["template_coeff"] = det_coeff

    def test_projection_unity(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create a default noise model
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Simulate some TOD
        ops.SimNoise(noise_model="noise_model").apply(data)

        ops.Copy(detdata=[("signal", "input")]).apply(data)

        # Create and apply fake templates
        self.fake_templates(data, "templates")

        ops.Copy(detdata=[("signal", "combined")]).apply(data)
        ops.Reset(detdata=["signal"]).apply(data)

        # Destriping template
        tmpl = PreComputed(obs_key="templates")

        # Set the data and initialize
        tmpl.data = data
        tmpl.initialize()

        # Get some amplitudes and set to one
        amps = tmpl.zeros()
        amps.local[:] = 1.0

        # Project.
        for det in tmpl.detectors():
            for ob in data.obs:
                tmpl.add_to_signal(det, amps)

        # Verify
        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                ddata = ob.detdata[defaults.det_data][det]
                dkey = ob["templates"]["det_to_key"][det]
                check = ob["templates"][dkey]
                if not np.allclose(ddata, check):
                    msg = f"FAIL: {ddata} != {check}"
                    print(msg, flush=True)
                    self.assertTrue(False)

        amps.reset()

        # Accumulate amplitudes
        for det in tmpl.detectors():
            for ob in data.obs:
                tmpl.project_signal(det, amps)

        # Verify- we have one amplitude per detector per observation
        offset = 0
        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                detamp = amps.local[offset]
                if not np.allclose(detamp, 1.0):
                    msg = f"FAIL: {detamp} != 1.0"
                    print(msg, flush=True)
                    self.assertTrue(False)
                offset += 1

        close_data(data)
