# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

import numpy as np
import numpy.testing as nt

from ..dist import distribute_uniform, distribute_discrete, Data
from ..mpi import Comm, MPI

from ._helpers import create_outdir, create_distdata


class DataTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # Create one observation per group.

        self.data = create_distdata(self.comm, obs_per_group=1)

        self.ntask = 24
        self.sizes1 = [
            29218,
            430879,
            43684,
            430338,
            36289,
            437553,
            37461,
            436200,
            41249,
            432593,
            42467,
            431195,
            35387,
            438274,
            36740,
            436741,
            40663,
            432999,
            42015,
            431285,
            35297,
            438004,
            37010,
            436291,
            41114,
            432186,
            42828,
            430293,
            36243,
            436697,
            38318,
            434802,
            42602,
            430338,
            44676,
            428264,
            38273,
            434306,
            40708,
            432051,
            45308,
            427452,
            36695,
            435884,
            41520,
            430879,
            44090,
            428309,
            38273,
            434126,
            40843,
            431375,
        ]
        self.totsamp1 = np.sum(self.sizes1)

        self.sizes2 = [(int(3600 * 169.7)) for i in range(8640)]
        self.totsamp2 = np.sum(self.sizes2)

    def test_construction(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        dist_uni1 = distribute_uniform(self.totsamp1, self.ntask)
        # with open("test_uni_{}".format(self.comm.rank), "w") as f:
        #     for d in dist_uni:
        #         f.write("uniform:  {} {}\n".format(d[0], d[1]))
        n1 = np.sum(np.array(dist_uni1)[:, 1])
        assert n1 == self.totsamp1

        n = self.totsamp1
        breaks = [n // 2 + 1000, n // 4 - 1000000, n // 2 + 1000, (3 * n) // 4]
        dist_uni2 = distribute_uniform(self.totsamp1, self.ntask, breaks=breaks)

        n2 = np.sum(np.array(dist_uni2)[:, 1])
        assert n2 == self.totsamp1

        for offset, nsamp in dist_uni2:
            for brk in breaks:
                if brk > offset and brk < offset + nsamp:
                    raise Exception(
                        "Uniform data distribution did not honor the breaks"
                    )

        dist_disc1 = distribute_discrete(self.sizes1, self.ntask)
        # with open("test_disc_{}".format(self.comm.rank), "w") as f:
        #     for d in dist_disc:
        #         f.write("discrete:  {} {}\n".format(d[0], d[1]))

        n = np.sum(np.array(dist_disc1)[:, 1])
        assert n == len(self.sizes1)

        n = len(self.sizes1)
        breaks = [n // 2, n // 4, n // 2, (3 * n) // 4]
        dist_disc2 = distribute_discrete(self.sizes1, self.ntask, breaks=breaks)

        n = np.sum(np.array(dist_disc2)[:, 1])
        assert n == len(self.sizes1)

        for offset, nchunk in dist_disc2:
            for brk in breaks:
                if brk > offset and brk < offset + nchunk:
                    raise Exception(
                        "Discrete data distribution did not honor the breaks"
                    )

        handle = None
        if rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_construct_info"), "w")
        self.data.info(handle)
        if rank == 0:
            handle.close()

        dist_disc3 = distribute_discrete(self.sizes2, 384)

        if rank == 0:
            with open(
                os.path.join(self.outdir, "dist_discrete_8640x384.txt"), "w"
            ) as f:
                indx = 0
                for d in dist_disc3:
                    f.write("{:04d} = ({}, {})\n".format(indx, d[0], d[1]))
                    indx += 1
        return

    def test_split(self):
        data = Data(self.data.comm)
        data.obs.append({"site": "Atacama", "season": 1})
        data.obs.append({"site": "Atacama", "season": 2})
        data.obs.append({"site": "Atacama", "season": 3})
        data.obs.append({"site": "Pole", "season": 1})
        data.obs.append({"site": "Pole", "season": 2})
        data.obs.append({"site": "Pole", "season": 3})

        datasplit_site = data.split("site")
        datasplit_season = data.split("season")

        nt.assert_equal(len(datasplit_site), 2)
        nt.assert_equal(len(datasplit_season), 3)

        # Verify that the observations are shared

        sum1 = 0
        for value, site_data in datasplit_site:
            for obs in site_data.obs:
                assert "var1" not in obs
                obs["var1"] = 1
                sum1 += 1

        sum2 = 0
        for value, season_data in datasplit_season:
            for obs in season_data.obs:
                sum2 += obs["var1"]

        nt.assert_equal(sum1, sum2)
        return

    def test_none(self):
        # test that Comm with None argument returns a None communicator
        if MPI is None:
            comm = Comm(None)
        else:
            comm = Comm(MPI.COMM_SELF)
        assert comm.comm_world is None
