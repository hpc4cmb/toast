# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

import numpy as np

from ..tod import OpPolyFilter, AnalyticNoise, OpSimNoise, Interval
from ..todmap import TODHpixSpiral

from ._helpers import create_outdir, create_distdata, boresight_focalplane


class OpPolyFilterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # Create one observation per group, and each observation will have
        # one detector per process and a single chunk.  Data within an
        # observation is distributed by detector.

        self.data = create_distdata(self.comm, obs_per_group=1)
        self.ndet = self.data.comm.group_size
        self.rate = 20.0

        # Create detectors with a range of knee frequencies.
        (
            dnames,
            dquat,
            depsilon,
            drate,
            dnet,
            dfmin,
            dfknee,
            dalpha,
        ) = boresight_focalplane(
            self.ndet,
            samplerate=self.rate,
            net=10.0,
            fmin=1.0e-5,
            fknee=np.linspace(0.01, 0.19, num=self.ndet),
        )

        # Fitting order
        self.order = 5

        # Total Samples
        self.totsamp = 100000

        # Populate the observations (one per group)

        tod = TODHpixSpiral(
            self.data.comm.comm_group,
            dquat,
            self.totsamp,
            detranks=self.data.comm.group_size,
            firsttime=0.0,
            rate=self.rate,
            nside=512,
        )

        # Construct an analytic noise model for the detectors

        nse = AnalyticNoise(
            rate=drate,
            fmin=dfmin,
            detectors=dnames,
            fknee=dfknee,
            alpha=dalpha,
            NET=dnet,
        )

        intervals = []
        interval_len = 1000
        for istart in range(0, self.totsamp, interval_len):
            istop = min(istart + interval_len, self.totsamp)
            intervals.append(
                Interval(
                    start=istart / self.rate,
                    stop=istop / self.rate,
                    first=istart,
                    last=istop - 1,
                )
            )

        self.data.obs[0]["tod"] = tod
        self.data.obs[0]["noise"] = nse
        self.data.obs[0]["intervals"] = intervals

    def test_filter(self):
        # generate timestreams
        op = OpSimNoise()
        op.exec(self.data)

        # Replace the noise with a polynomial fit
        old_rms = []
        for ob in self.data.obs:
            tod = ob["tod"]
            orms = {}
            for det in tod.local_dets:
                cachename = "noise_{}".format(det)
                y = tod.cache.reference(cachename)
                x = np.arange(y.size)
                p = np.polyfit(x, y, self.order)
                y[:] = np.polyval(p, x)
                orms[det] = np.std(y)
                del y
            old_rms.append(orms)

        # Filter timestreams
        op = OpPolyFilter(name="noise", order=self.order)
        op.exec(self.data)

        # Ensure all timestreams are zeroed out by the filter.
        # The polynomial basis used in populating the TOD is different
        # than the one filter uses and the basis functions are not
        # strictly orthogonal on a sparse grid so we expect a low level
        # residual.

        for ob, orms in zip(self.data.obs, old_rms):
            tod = ob["tod"]
            for det in tod.local_dets:
                cachename = "noise_{}".format(det)
                y = tod.cache.reference(cachename)
                rms = np.std(y)
                old = orms[det]
                if np.abs(rms / old) > 1e-6:
                    raise RuntimeError(
                        "det {} old rms = {}, new rms = {}" "".format(det, old, rms)
                    )
                del y
        return
