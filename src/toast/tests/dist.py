# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re

import numpy as np
import numpy.testing as nt
from astropy import units as u

from ..data import Data
from ..dist import distribute_discrete, distribute_uniform
from ..instrument import Session
from ..mpi import MPI, Comm
from ..observation import Observation
from ._helpers import (
    close_data,
    create_comm,
    create_ground_telescope,
    create_outdir,
    create_satellite_data,
    create_satellite_empty,
)
from .mpi import MPITestCase


class DataTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # Create one observation per group.

        self.data = create_satellite_empty(self.comm, obs_per_group=1)

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
        self.data.info(handle=handle)
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

    def test_view(self):
        data = create_satellite_data(
            self.comm, obs_per_group=1, obs_time=10.0 * u.minute
        )
        # Add some metadata to the data dictionary and every observation
        data["meta1"] = "foo"
        data["meta2"] = "bar"
        for ob in data.obs:
            ob["special"] = "value"

        # Select all observations, which makes a data view that has a reference to
        # the metadata and all observations
        alt_data = data.select(obs_key="special")

        # Verify that changing metadata modifies the original
        alt_data["meta1"] = "blat"
        self.assertTrue(data["meta1"] == "blat")

        # Verify that deleting our selection does not clear the original
        alt_data.clear()
        del alt_data
        self.assertTrue("boresight_radec" in data.obs[0].shared)
        close_data(data)

    def test_select(self):
        toastcomm = create_comm(self.comm)
        tele = create_ground_telescope(toastcomm.group_size)
        data = Data(toastcomm)
        get_uid = None
        for season in range(3):
            data.obs.append(
                Observation(toastcomm, tele, 10, name=f"atacama-{season:02d}")
            )
            data.obs[-1]["site"] = "Atacama"
            data.obs[-1]["season"] = season
            get_uid = data.obs[-1].uid
        for season in range(3):
            data.obs.append(Observation(toastcomm, tele, 10, name=f"pole-{season:02d}"))
            data.obs[-1]["site"] = "Pole"
            data.obs[-1]["season"] = season

        selected_indx = data.select(obs_index=1)
        self.assertTrue(len(selected_indx.obs) == 1)

        selected_uid = data.select(obs_uid=get_uid)
        self.assertTrue(len(selected_uid.obs) == 1)

        name_pat = re.compile(r"pole-.*")
        selected_namepat = data.select(obs_name=name_pat)
        self.assertTrue(len(selected_namepat.obs) == 3)

        selected_name = data.select(obs_name="atacama-00")
        self.assertTrue(len(selected_name.obs) == 1)

        selected_key = data.select(obs_key="season")
        self.assertTrue(len(selected_key.obs) == 6)

        selected_keyval = data.select(obs_key="season", obs_val=1)
        self.assertTrue(len(selected_keyval.obs) == 2)
        close_data(data)

    def test_split(self):
        toastcomm = create_comm(self.comm)
        tele = create_ground_telescope(toastcomm.group_size)
        data = Data(toastcomm)
        for season in range(3):
            data.obs.append(
                Observation(toastcomm, tele, 10, name=f"atacama-{season:02d}")
            )
            data.obs[-1]["site"] = "Atacama"
            data.obs[-1]["season"] = season
        for season in range(3):
            data.obs.append(Observation(toastcomm, tele, 10, name=f"pole-{season:02d}"))
            data.obs[-1]["site"] = "Pole"
            data.obs[-1]["season"] = season

        datasplit_indx = data.split(obs_index=True, require_full=True)
        self.assertTrue(len(datasplit_indx) == 6)

        datasplit_name = data.split(obs_name=True, require_full=True)
        self.assertTrue(len(datasplit_name) == 6)

        datasplit_uid = data.split(obs_uid=True, require_full=True)
        self.assertTrue(len(datasplit_uid) == 6)

        datasplit_site = data.split(obs_key="site")
        datasplit_season = data.split(obs_key="season")

        self.assertTrue(len(datasplit_site) == 2)
        self.assertTrue(len(datasplit_season) == 3)

        # Verify that the observations are shared

        sum1 = 0
        for value, site_data in datasplit_site.items():
            for obs in site_data.obs:
                assert "var1" not in obs
                obs["var1"] = 1
                sum1 += 1

        sum2 = 0
        for value, season_data in datasplit_season.items():
            for obs in season_data.obs:
                sum2 += obs["var1"]

        nt.assert_equal(sum1, sum2)
        del datasplit_indx
        del datasplit_name
        del datasplit_season
        del datasplit_site
        del datasplit_uid
        close_data(data)

    def test_none(self):
        # test that Comm with None argument returns a None communicator
        if MPI is None:
            comm = Comm(None)
        else:
            comm = Comm(MPI.COMM_SELF)
        assert comm.comm_world is None

    def test_session(self):
        toastcomm = create_comm(self.comm)
        tele = create_ground_telescope(toastcomm.group_size)
        data = Data(toastcomm)
        get_uid = None
        # Note:  for testing we are just re-using the "season"
        # as the "session", which is not physical.  This is just
        # for testing.
        for season in range(3):
            data.obs.append(
                Observation(
                    toastcomm,
                    tele,
                    10,
                    name=f"atacama-{season:02d}",
                    session=Session(
                        f"{season:02d}",
                    ),
                )
            )
            data.obs[-1]["site"] = "Atacama"
            data.obs[-1]["season"] = season
            get_uid = data.obs[-1].uid
        for season in range(3):
            data.obs.append(
                Observation(
                    toastcomm,
                    tele,
                    10,
                    name=f"pole-{season:02d}",
                    session=Session(
                        f"{season:02d}",
                    ),
                )
            )
            data.obs[-1]["site"] = "Pole"
            data.obs[-1]["season"] = season

        datasplit_session = data.split(obs_session_name=True)
        self.assertTrue(len(datasplit_session) == 3)
        for skey in datasplit_session.keys():
            self.assertTrue(len(datasplit_session[skey].obs) == 2)
            for ob in datasplit_session[skey].obs:
                mat = re.match(r".*-(\d\d)", ob.name)
                self.assertTrue(mat.group(1) == skey)

        for season in range(3):
            sname = f"{season:02d}"
            sel = data.select(obs_session_name=sname)
            self.assertTrue(len(sel.obs) == 2)
            for ob in sel.obs:
                mat = re.match(r".*-(\d\d)", ob.name)
                self.assertTrue(mat.group(1) == sname)

        del datasplit_session
        close_data(data)
