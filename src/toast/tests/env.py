# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

from ..utils import Environment

from ._helpers import create_comm


class EnvTest(MPITestCase):
    def setUp(self):
        self.rank = 0
        self.nproc = 1
        if self.comm is not None:
            self.rank = self.comm.rank
            self.nproc = self.comm.size

    def test_env(self):
        env = Environment.get()
        if self.rank == 0:
            print(env, flush=True)

    def test_comm(self):
        comm = create_comm(self.comm)
        for p in range(self.nproc):
            if p == self.rank:
                print(comm, flush=True)
            if self.comm is not None:
                self.comm.barrier()
