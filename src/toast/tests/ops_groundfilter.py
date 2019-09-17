# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

import numpy as np

from ..tod import AnalyticNoise, OpSimNoise
from ..todmap import OpGroundFilter, TODGround

from ._helpers import create_outdir, create_distdata, boresight_focalplane


class OpGroundFilterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # One observation per group
        self.data = create_distdata(self.comm, obs_per_group=1)

        # Detector properties.  We place one detector per process at the
        # boresight with evenly spaced polarization orientations.
        self.ndet = self.data.comm.group_size
        self.NET = 10.0
        self.rate = 20.0

        # Create detectors with a range of knee frequencies.
        dnames, dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha = boresight_focalplane(
            self.ndet,
            samplerate=self.rate,
            net=self.NET,
            fmin=1.0e-5,
            fknee=np.linspace(0.0, 0.1, num=self.ndet),
        )

        # Samples per observation
        self.totsamp = 100000

        # bin size
        self.order = 3

        # Populate the observation
        tod = TODGround(
            self.data.comm.comm_group,
            dquat,
            self.totsamp,
            detranks=self.data.comm.group_size,
            firsttime=0.0,
            rate=self.rate,
            azmin=45,
            azmax=55,
            el=45,
        )

        # construct an analytic noise model
        nse = AnalyticNoise(
            rate=drate,
            fmin=dfmin,
            detectors=dnames,
            fknee=dfknee,
            alpha=dalpha,
            NET=dnet,
        )

        self.data.obs[0]["tod"] = tod
        self.data.obs[0]["noise"] = nse

    def test_filter(self):
        # generate timestreams
        op = OpSimNoise()
        op.exec(self.data)

        # Replace the noise with a ground-synchronous signal
        old_rms = []
        for ob in self.data.obs:
            tod = ob["tod"]
            az = tod.read_boresight_az()
            orms = {}
            for det in tod.local_dets:
                cachename = "noise_{}".format(det)
                y = tod.cache.reference(cachename)
                y[:] = np.sin(az)
                orms[det] = np.std(y)
                del y
            old_rms.append(orms)

        # Filter timestreams
        op = OpGroundFilter(name="noise", filter_order=self.order, common_flag_mask=0)
        op.exec(self.data)

        # Ensure all timestreams are zeroed out by the filter
        for ob, orms in zip(self.data.obs, old_rms):
            tod = ob["tod"]
            for det in tod.local_dets:
                cachename = "noise_{}".format(det)
                y = tod.cache.reference(cachename)
                rms = np.std(y)
                old = orms[det]
                if np.abs(rms / old) > 1.0e-3:
                    raise RuntimeError(
                        "det {} old rms = {}, new rms = {}" "".format(det, old, rms)
                    )
                del y
        return
