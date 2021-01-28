# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from astropy import units as u

from .mpi import MPITestCase

from ..utils import rate_from_times

from ..dist import distribute_uniform

from .. import ops

from ..templates import Amplitudes, AmplitudesMap

from ._helpers import create_outdir, create_comm


class TemplateTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_amplitudes(self):
        # Create a toast communicator with groups if possible
        comm = create_comm(self.comm)

        # Global number of amplitudes
        n_global = 1000

        # Test every process with independent amplitudes

        dist = distribute_uniform(n_global, comm.world_size)
        n_local = dist[comm.world_rank][1]

        amps = Amplitudes(comm.comm_world, n_global, n_local, dtype=np.int32)

        amps.local[:] = 1
        amps.sync()

        np.testing.assert_equal(amps.local, np.ones_like(amps.local))

        amps.clear()
        del amps

        # Test every process with full overlap

        n_local = n_global
        amps = Amplitudes(comm.comm_world, n_global, n_local, dtype=np.int32)
        amps.local[:] = 1
        amps.sync()

        np.testing.assert_equal(amps.local, comm.world_size * np.ones_like(amps.local))

        amps.clear()
        del amps

        # Test arbitrary distribution

        n_local = n_global // 2
        local_indices = 2 * np.arange(n_local, dtype=np.int32)
        local_indices += comm.world_rank % 2

        amps = Amplitudes(
            comm.comm_world,
            n_global,
            n_local,
            local_indices=local_indices,
            dtype=np.int32,
        )
        amps.local[:] = 1
        amps.sync()

        check_even = (1 + comm.world_size) // 2
        check_odd = comm.world_size // 2

        if comm.world_rank % 2 == 0:
            np.testing.assert_equal(amps.local, check_even * np.ones_like(amps.local))
        else:
            np.testing.assert_equal(amps.local, check_odd * np.ones_like(amps.local))

        amps.clear()
        del amps

        return
