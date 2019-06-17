# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

from ..tod import TODHpixSpiral, OpPointingHpix

from ..map import OpLocalPixels

from ._helpers import create_outdir, create_distdata, boresight_focalplane


class OpPointingHpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # Create one observation per group, and each observation will have
        # one detector per process and a single chunk.  Data within an
        # observation is distributed by detector.

        self.data = create_distdata(self.comm, obs_per_group=1)
        self.ndet = self.data.comm.group_size

        # Create detectors with default properties
        dnames, dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha = boresight_focalplane(
            self.ndet
        )

        # A small number of samples
        self.totsamp = 10

        # Populate the observations (one per group)

        tod = TODHpixSpiral(
            self.data.comm.comm_group,
            dquat,
            self.totsamp,
            detranks=self.data.comm.comm_group.size,
        )

        self.data.obs[0]["tod"] = tod

    def tearDown(self):
        del self.data

    def test_hpix_simple(self):
        op = OpPointingHpix()
        op.exec(self.data)

        lc = OpLocalPixels()
        local = lc.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_hpix_simple_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()
        return

    def test_hpix_hwpnull(self):
        op = OpPointingHpix(mode="IQU")
        op.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_hpix_hwpnull_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()
        return
