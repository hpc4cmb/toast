# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .. import ops as ops
from ..observation import default_values as defaults
from .helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class DetrendTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def _test_detrend(self, method):

        data = create_ground_data(self.comm)

        # Flag some samples
        for ob in data.obs:
            nsample = ob.shared[defaults.times].data.shape[0]
            for i, det in enumerate(ob.local_detectors):
                detflags = ob.detdata[defaults.det_flags][det]
                detflags[:] = 0
                # Flag some samples at the edge
                edge_nsample = i
                if i > 0:
                    detflags[:edge_nsample] = defaults.det_mask_invalid
                    detflags[-edge_nsample:] = defaults.det_mask_invalid
                # Flag some samples at the middle
                for j in range(
                        edge_nsample+len(ob.local_detectors)+100,
                        nsample-len(ob.local_detectors)-edge_nsample-100,
                        200):
                    detflags[j+j%10:j+10*(j%10)] = defaults.det_mask_invalid
                # Flag a large chunk of data
                if nsample//4 > edge_nsample+len(ob.local_detectors)+100:
                    detflags[nsample//4:nsample//2] = defaults.det_mask_invalid

        if method == 'linear':
            nsample0 = max(*[ob.shared[defaults.times].data.shape[0] for ob in data.obs])
            # Test edge samples cases:
            # - 1 edge samples
            # - odd/even edge samples
            # - edge samples is half of total samples
            # - edge samples is total samples
            # - edge samples is larger than total samples
            for edge_nsample in [1,2,3,4,5, nsample0//2, nsample0, nsample0+100]:
                abnormal_number = edge_nsample%10 * (-1)**(edge_nsample%2) * 1e5
                for ob in data.obs:
                    nsample = ob.shared[defaults.times].data.shape[0]
                    for i, det in enumerate(ob.local_detectors):
                        detflags = ob.detdata[defaults.det_flags][det]
                        detdata = ob.detdata[defaults.det_data][det]
                        detdata[:] = (-1)**(i//2) * np.arange(float(nsample))
                        # For all flagged samples add some value to detect if they are ignored
                        detdata[detflags != 0] += abnormal_number

                linear_detrend = ops.Detrend(
                        method='linear',
                        edge_nsample=edge_nsample,
                        edge_nsample_method=['mean', 'median'][edge_nsample%2],
                        )
                linear_detrend.apply(data)

                for ob in data.obs:
                    nsample = ob.shared[defaults.times].data.shape[0]
                    if edge_nsample in [nsample0//2, nsample0, nsample0+100]:
                        assert len(ob.select_local_detectors(flagmask=defaults.det_mask_invalid)) == 0
                    for i, det in enumerate(ob.select_local_detectors(flagmask=defaults.det_mask_invalid)):
                        detdata = ob.detdata[defaults.det_data][det]
                        detflags = ob.detdata[defaults.det_flags][det]
                        detdata[detflags != 0] -= abnormal_number
                        if edge_nsample < nsample0//2:
                            assert np.all(detdata == 0.0)
                        else:
                            assert np.allclose(np.abs(detdata), np.arange(float(nsample)))

        else:
            for ob in data.obs:
                nsample = ob.shared[defaults.times].data.shape[0]
                for i, det in enumerate(ob.local_detectors):
                    detflags = ob.detdata[defaults.det_flags][det]
                    detdata = ob.detdata[defaults.det_data][det]
                    detdata[:] = (-1)**(i//2) * np.arange(float(nsample))
                    # multiply number of unflagged samples reduce rounding error when calculating mean
                    detdata *= np.sum(detflags == 0)
                    # For all flagged samples add some value to detect if they are ignored
                    detdata[detflags != 0] += 1e10

            detrend = ops.Detrend(method=method)
            detrend.apply(data)

            for ob in data.obs:
                for i, det in enumerate(ob.select_local_detectors(flagmask=defaults.det_mask_invalid)):
                    detdata = ob.detdata[defaults.det_data][det]
                    detflags = ob.detdata[defaults.det_flags][det]
                    if method == 'mean':
                        assert np.mean(detdata[detflags==0]) == 0.0
                    elif method == 'median':
                        assert np.median(detdata[detflags==0]) == 0.0
                    else:
                        raise RuntimeError(f"Unknown method={method}")

        close_data(data)

    def test_detrend_median(self):
        self._test_detrend(method='median')

    def test_detrend_mean(self):
        self._test_detrend(method='mean')

    def test_detrend_linear(self):
        self._test_detrend(method='linear')

