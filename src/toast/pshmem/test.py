##
# Copyright (c) 2017-2020, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

import os
import sys
import time

import unittest

import numpy as np
import numpy.testing as nt

from .shmem import MPIShared
from .locking import MPILock

MPI = None
use_mpi = True

if "PSHMEM_MPI_DISABLE" in os.environ:
    use_mpi = False

if use_mpi and (MPI is None):
    try:
        import mpi4py.MPI as MPI
    except ImportError:
        raise ImportError("Cannot import mpi4py, will only test serial functionality.")


class ShmemTest(unittest.TestCase):
    def setUp(self):
        self.comm = None
        if MPI is not None:
            self.comm = MPI.COMM_WORLD
        self.rank = 0
        self.procs = 1
        if self.comm is not None:
            self.rank = self.comm.rank
            self.procs = self.comm.size

    def tearDown(self):
        pass

    def test_allocate(self):
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

        for datatype in [np.int32, np.int64, np.float32, np.float64]:

            # For testing the "set()" method, every process is going to
            # create a full-sized data buffer and fill it with its process rank.
            local = np.ones(datadims, dtype=datatype)
            local *= self.rank

            # A context manager is the pythonic way to make sure that the
            # object has no dangling reference counts after leaving the context,
            # and will ensure that the shared memory is freed properly.

            with MPIShared(local.shape, local.dtype, self.comm) as shm:
                for p in range(self.procs):
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
                                "proc {} threw exception during set()".format(rank),
                                flush=True,
                            )
                            if self.comm is not None:
                                self.comm.Abort()
                            else:
                                sys.exit(1)

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

                for p in range(self.procs):
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

    def test_shape(self):
        good_dims = [
            (2, 5, 10),
            np.array([10, 2], dtype=np.int32),
            np.array([5, 2], dtype=np.int64),
            np.array([10, 2], dtype=np.int),
        ]
        bad_dims = [
            (2, 5.5, 10),
            np.array([10, 2], dtype=np.float32),
            np.array([5, 2], dtype=np.float64),
            np.array([10, 2.5], dtype=np.float32),
        ]

        dt = np.float64

        for dims in good_dims:
            try:
                shm = MPIShared(dims, dt, self.comm)
                if self.rank == 0:
                    print("successful creation with shape {}".format(dims), flush=True)
                del shm
            except ValueError:
                if self.rank == 0:
                    print(
                        "unsuccessful creation with shape {}".format(dims), flush=True
                    )
        for dims in bad_dims:
            try:
                shm = MPIShared(dims, dt, self.comm)
                if self.rank == 0:
                    print("unsuccessful rejection of shape {}".format(dims), flush=True)
                del shm
            except ValueError:
                if self.rank == 0:
                    print("successful rejection of shape {}".format(dims), flush=True)


class LockTest(unittest.TestCase):
    def setUp(self):
        self.comm = None
        if MPI is not None:
            self.comm = MPI.COMM_WORLD
        self.rank = 0
        self.procs = 1
        if self.comm is not None:
            self.rank = self.comm.rank
            self.procs = self.comm.size
        self.sleepsec = 0.2

    def tearDown(self):
        pass

    def test_lock(self):
        with MPILock(self.comm, root=0, debug=True) as lock:
            for lk in range(5):
                msg = "test_lock:  process {} got lock {}".format(
                    self.rank, lk
                )
                lock.lock()
                print(msg, flush=True)
                #time.sleep(self.sleepsec)
                lock.unlock()
        if self.comm is not None:
            self.comm.barrier()


def run():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(LockTest))
    suite.addTest(unittest.makeSuite(ShmemTest))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    return
