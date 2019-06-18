# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

import numpy as np

from ..tod import OpGainScrambler, TODHpixSpiral, AnalyticNoise, OpSimNoise

from ._helpers import (
    create_outdir,
    create_distdata,
    boresight_focalplane,
    uniform_chunks,
)


class OpGainScramblerTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # One observation per group
        self.data = create_distdata(self.comm, obs_per_group=1)

        self.ndet = 4
        self.rate = 20.0

        # Create detectors with a range of knee frequencies.
        dnames, dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha = boresight_focalplane(
            self.ndet,
            samplerate=self.rate,
            net=10.0,
            fmin=1.0e-5,
            fknee=np.linspace(0.0, 0.1, num=self.ndet),
        )

        # Total samples per observation
        self.totsamp = 200000

        # Chunks
        chunks = uniform_chunks(self.totsamp, nchunk=self.data.comm.group_size)

        # Construct an empty TOD (no pointing needed)

        tod = TODHpixSpiral(
            self.data.comm.comm_group,
            dquat,
            self.totsamp,
            firsttime=0.0,
            rate=self.rate,
            nside=512,
            sampsizes=chunks,
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

    def test_scrambler(self):
        # generate timestreams
        op = OpSimNoise()
        op.exec(self.data)

        # Record the old RMS
        old_rms = []
        for ob in self.data.obs:
            tod = ob["tod"]
            orms = {}
            for det in tod.local_dets:
                cachename = "noise_{}".format(det)
                y = tod.cache.reference(cachename)
                orms[det] = np.std(y)
                del y
            old_rms.append(orms)

        # Scramble the timestreams

        op = OpGainScrambler(center=2, sigma=1e-6, name="noise")
        op.exec(self.data)

        # Ensure RMS changes for the implicated detectors

        for ob, orms in zip(self.data.obs, old_rms):
            tod = ob["tod"]
            for det in tod.local_dets:
                cachename = "noise_{}".format(det)
                y = tod.cache.reference(cachename)
                rms = np.std(y)
                old = orms[det]
                if np.abs(rms / old) - 2 > 1e-3:
                    raise RuntimeError(
                        "det {} old rms = {}, new rms = {}".format(det, old, rms)
                    )
                del y

        return
