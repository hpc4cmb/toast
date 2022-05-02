# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import time

import numpy as np
import numpy.testing as nt

from ..mpi import MPILock, MPIShared
from ..utils import Environment
from ._helpers import create_comm
from .mpi import MPITestCase


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

    def test_mpi_shared(self):
        """This is based on the simple tests from the upstream repo:
        https://github.com/tskisner/mpi_shmem/blob/master/test.py
        """
        # Dimensions of our shared memory array
        datadims = (2, 5, 10)

        # Dimensions of the incremental slab that we will
        # copy during each set() call.
        updatedims = (1, 1, 5)

        # How many updates are there to cover the whole
        # data array?
        nupdate = 1
        for d in range(len(datadims)):
            nupdate *= datadims[d] // updatedims[d]

        for datatype in [np.float64, np.float32, np.int64, np.int32]:
            # For testing the "set()" method, every process is going to
            # create a full-sized data buffer and fill it with its process rank.
            local = np.ones(datadims, dtype=datatype)
            local *= self.rank

            with MPIShared(local.shape, local.dtype, self.comm) as shm:
                for p in range(self.nproc):
                    # Every process takes turns writing to the buffer.
                    setdata = None
                    setoffset = (0, 0, 0)

                    # Write to the whole data volume, but in small blocks
                    for upd in range(nupdate):
                        if p == self.rank:
                            # My turn!  Write my process rank to the buffer slab.
                            setdata = local[
                                setoffset[0] : setoffset[0] + updatedims[0],
                                setoffset[1] : setoffset[1] + updatedims[1],
                                setoffset[2] : setoffset[2] + updatedims[2],
                            ]
                        try:
                            # All processes call set(), but only data on rank p matters.
                            shm.set(setdata, setoffset, fromrank=p)
                        except:
                            print(
                                "proc {} threw exception during set()".format(
                                    self.rank
                                ),
                                flush=True,
                            )
                            raise RuntimeError("shared memory set failed")

                        # Increment the write offset within the array

                        x = setoffset[0]
                        y = setoffset[1]
                        z = setoffset[2]

                        z += updatedims[2]
                        if z >= datadims[2]:
                            z = 0
                            y += updatedims[1]
                        if y >= datadims[1]:
                            y = 0
                            x += updatedims[0]

                        setoffset = (x, y, z)

                    # Every process is now going to read a copy from the shared memory
                    # and make sure that they see the data written by the current process.
                    check = np.zeros_like(local)
                    check[:, :, :] = shm[:, :, :]

                    truth = np.ones_like(local)
                    truth *= p

                    # This should be bitwise identical, even for floats
                    nt.assert_equal(check[:, :, :], truth[:, :, :])

                # Ensure that we can reference the memory buffer from numpy without
                # a memory copy.  The intention is that a slice of the shared memory
                # buffer should appear as a C-contiguous ndarray whenever we slice
                # along the last dimension.

                for p in range(self.nproc):
                    if p == self.rank:
                        slc = shm[1, 2]
                        print(
                            "proc {} slice has dims {}, dtype {}, C = {}".format(
                                p, slc.shape, slc.dtype.str, slc.flags["C_CONTIGUOUS"]
                            ),
                            flush=True,
                        )
                    if self.comm is not None:
                        self.comm.barrier()

    def test_mpi_lock(self):
        """This is based on the simple tests from the upstream repo:
        https://github.com/tskisner/mpi_shmem/blob/master/test.py
        """
        sleepsec = 0.1
        lock = MPILock(self.comm, root=0, debug=True)

        msg = "test lock:  process {} got the lock".format(self.rank)

        lock.lock()
        print(msg, flush=True)
        time.sleep(sleepsec)
        lock.unlock()

        if self.comm is not None:
            self.comm.barrier()
