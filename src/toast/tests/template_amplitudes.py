# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops
from ..dist import distribute_uniform
from ..templates import Amplitudes, AmplitudesMap
from ..utils import rate_from_times
from ._helpers import create_comm, create_outdir
from .mpi import MPITestCase


class TemplateTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_amplitudes_disjoint(self):
        # Create a toast communicator with groups if possible
        comm = create_comm(self.comm)

        # Global number of amplitudes
        n_global = 1000

        # Test every process with independent amplitudes

        dist = distribute_uniform(n_global, comm.world_size)
        n_local = dist[comm.world_rank][1]

        for cbytes in [500, 1000000]:
            amps = Amplitudes(comm, n_global, n_local, dtype=np.int32)

            amps.local[:] = 1
            amps.sync(comm_bytes=cbytes)

            np.testing.assert_equal(amps.local, np.ones_like(amps.local))

            dup = amps.duplicate()
            cdot = dup.dot(amps, comm_bytes=cbytes)
            np.testing.assert_equal(cdot, n_global)

    def test_amplitudes_full(self):
        # Create a toast communicator with groups if possible
        comm = create_comm(self.comm)

        # Global number of amplitudes
        n_global = 1000

        # Test every process with full overlap

        n_local = n_global

        for cbytes in [500, 1000000]:
            amps = Amplitudes(comm, n_global, n_local, dtype=np.int32)
            amps.local[:] = 1
            amps.sync(comm_bytes=cbytes)

            np.testing.assert_equal(
                amps.local, comm.world_size * np.ones_like(amps.local)
            )

            dup = amps.duplicate()
            cdot = dup.dot(amps, comm_bytes=cbytes)
            np.testing.assert_equal(cdot, (comm.world_size**2) * n_global)

    def test_amplitudes_range(self):
        # Create a toast communicator with groups if possible
        comm = create_comm(self.comm)

        # Global number of amplitudes
        n_global = 1000

        # Test overlapping ranges

        rlen = 50
        rskip = 2 * rlen
        nr = n_global // rskip

        ranges_a = list()
        ranges_b = list()
        for i in range(nr):
            ranges_a.append((rskip * i, rlen))
            ranges_b.append((rlen + rskip * i, rlen))

        n_local = nr * rlen
        lranges = None
        if comm.world_rank % 2 == 0:
            lranges = ranges_a
        else:
            lranges = ranges_b

        for cbytes in [500, 1000000]:
            amps = Amplitudes(
                comm,
                n_global,
                n_local,
                local_ranges=lranges,
                dtype=np.int32,
            )
            amps.local[:] = 1
            amps.sync(comm_bytes=cbytes)

            check_even = (1 + comm.world_size) // 2
            check_odd = comm.world_size // 2

            if comm.world_rank % 2 == 0:
                np.testing.assert_equal(
                    amps.local, check_even * np.ones_like(amps.local)
                )
            else:
                np.testing.assert_equal(
                    amps.local, check_odd * np.ones_like(amps.local)
                )

            dup = amps.duplicate()
            cdot = dup.dot(amps, comm_bytes=cbytes)
            np.testing.assert_equal(
                cdot,
                (check_even**2 + check_odd**2) * n_global / 2,
            )

    def test_amplitudes_indexed(self):
        # Create a toast communicator with groups if possible
        comm = create_comm(self.comm)

        # Global number of amplitudes
        n_global = 1000

        # Test arbitrary distribution

        n_local = n_global // 2
        local_indices = 2 * np.arange(n_local, dtype=np.int32)
        local_indices += comm.world_rank % 2

        for cbytes in [500, 1000000]:
            amps = Amplitudes(
                comm,
                n_global,
                n_local,
                local_indices=local_indices,
                dtype=np.int32,
            )
            amps.local[:] = 1
            amps.sync(comm_bytes=cbytes)

            check_even = (1 + comm.world_size) // 2
            check_odd = comm.world_size // 2

            if comm.world_rank % 2 == 0:
                np.testing.assert_equal(
                    amps.local, check_even * np.ones_like(amps.local)
                )
            else:
                np.testing.assert_equal(
                    amps.local, check_odd * np.ones_like(amps.local)
                )

            dup = amps.duplicate()
            cdot = dup.dot(amps, comm_bytes=cbytes)
            np.testing.assert_equal(
                cdot, (check_even**2 + check_odd**2) * n_global / 2
            )

    def test_amplitudes_group(self):
        # Create a toast communicator with groups if possible
        comm = create_comm(self.comm)

        # Global number of amplitudes
        n_global = 1000

        # Test overlapping ranges

        rlen = 50
        rskip = 2 * rlen
        nr = n_global // rskip

        ranges_a = list()
        ranges_b = list()
        for i in range(nr):
            ranges_a.append((rskip * i, rlen))
            ranges_b.append((rlen + rskip * i, rlen))

        n_local = nr * rlen
        lranges = None
        if comm.world_rank % 2 == 0:
            lranges = ranges_a
        else:
            lranges = ranges_b

        for cbytes in [500, 1000000]:
            amps = Amplitudes(
                comm,
                n_global,
                n_local,
                local_ranges=lranges,
                dtype=np.int32,
                use_group=True,
            )
            amps.local[:] = 1
            amps.sync(comm_bytes=cbytes)

            check_even = (1 + comm.group_size) // 2
            check_odd = comm.group_size // 2

            if comm.group_rank % 2 == 0:
                np.testing.assert_equal(
                    amps.local, check_even * np.ones_like(amps.local)
                )
            else:
                np.testing.assert_equal(
                    amps.local, check_odd * np.ones_like(amps.local)
                )

            dup = amps.duplicate()
            cdot = dup.dot(amps, comm_bytes=cbytes)
            np.testing.assert_equal(
                cdot,
                (check_even**2 + check_odd**2) * n_global / 2,
            )
