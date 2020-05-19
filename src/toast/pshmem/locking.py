##
# Copyright (c) 2017-2020, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

import sys
import itertools

import numpy as np

from .utils import mpi_check_abort, mpi_data_type


class MPILock(object):
    """
    Implement a MUTEX lock with MPI one-sided operations.

    The lock is created across the given communicator.  This uses an array
    of bytes (one per process) to track which processes have requested the
    lock.  When a given process releases the lock, it passes it to the next
    process in order of request.

    Args:
        comm (MPI.Comm): the full communicator to use.
        root (int): the rank which stores the list of waiting processes.
        debug (bool): if True, print debugging info about the lock status.
    """

    # This creates a new integer for each time the class is instantiated.
    newid = next(itertools.count())

    def __init__(self, comm, root=0, debug=False):
        self._comm = comm
        self._root = root
        self._debug = debug

        # A unique tag for each instance of the class
        self._tag = MPILock.newid

        self._rank = 0
        self._procs = 1
        if self._comm is not None:
            self._rank = self._comm.rank
            self._procs = self._comm.size

        if self._rank == self._root:
            self._nlocal = self._procs
        else:
            self._nlocal = 0

        self._dtype = np.dtype(np.int32)

        # Local memory buffer
        self._waiting = np.zeros((self._procs,), dtype=self._dtype)
        self._send_token = self._rank * np.ones((1,), dtype=self._dtype)
        self._recv_token = np.zeros((1,), dtype=self._dtype)

        # Data type sizes
        self._dsize, self._mpitype = mpi_data_type(self._comm, self._dtype)

        # Allocate the shared memory buffer.

        self._win = None
        self._have_lock = False

        nbytes = self._nlocal * self._dsize

        if self._comm is not None:
            from mpi4py import MPI

            # Root allocates the buffer
            status = 0
            try:
                self._win = MPI.Win.Allocate(
                    nbytes, disp_unit=self._dsize, info=MPI.INFO_NULL, comm=self._comm
                )
            except:
                status = 1
            mpi_check_abort(self._comm, self._root, status, "memory allocation")

            if self._rank == self._root:
                # Root sets to zero
                self._win.Lock(self._root, MPI.LOCK_EXCLUSIVE)
                self._win.Put(
                    [self._waiting, self._procs, self._mpitype],
                    self._root,
                    target=[0, self._procs, self._mpitype],
                )
                self._win.Flush(self._root)
                self._win.Unlock(self._root)

        return

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
        return False

    def close(self):
        # Explicitly free the shared window
        if hasattr(self, "_win") and (self._win is not None):
            self._win.Free()
            self._win = None
        return

    @property
    def comm(self):
        """
        The communicator.
        """
        return self._comm

    def lock(self):
        """
        Request the lock and wait.

        This call blocks until lock is available.  Then it acquires
        the lock and returns.
        """
        # Do we already have the lock?
        if self._have_lock:
            return

        if self._comm is not None:
            from mpi4py import MPI

            # lock the window
            if self._debug:
                print(
                    "lock:  rank {}, instance {} locking shared window".format(
                        self._rank, self._tag
                    ),
                    flush=True,
                )
            self._win.Lock(self._root, MPI.LOCK_EXCLUSIVE)

            # Get a local copy of the buffer

            self._win.Get(
                [self._waiting, self._procs, self._mpitype],
                self._root,
                target=[0, self._procs, self._mpitype],
            )
            if self._debug:
                print(
                    "lock:  rank {}, instance {} list = {}".format(
                        self._rank, self._tag, self._waiting
                    ),
                    flush=True,
                )

            # Find the highest wait number in the list.  The location of this is the
            # process rank that will be sending us the token.

            wait_max = np.max(self._waiting)
            my_wait = (wait_max + 1) * np.ones((1,), dtype=self._dtype)

            sender = -1
            if wait_max > 0:
                sender = np.argmax(self._waiting)

            # Update the waiting list

            self._win.Put(
                [my_wait, 1, self._mpitype],
                self._root,
                target=[self._rank, 1, self._mpitype],
            )
            if self._debug:
                print(
                    "lock:  rank {}, instance {} putting wait number {}".format(
                        self._rank, self._tag, my_wait[0]
                    ),
                    flush=True,
                )

            # Flush

            self._win.Flush(self._root)

            # Release the window lock

            if self._debug:
                print(
                    "lock:  rank {}, instance {} unlocking shared window".format(
                        self._rank, self._tag
                    ),
                    flush=True,
                )
            self._win.Unlock(self._root)

            # If another rank has the token, wait for that

            if sender >= 0:
                if self._debug:
                    print(
                        "lock:  rank {} waiting for the lock from {}".format(
                            self._rank, sender
                        ),
                        flush=True,
                    )
                self._comm.Recv(self._recv_token, source=sender, tag=self._tag)

            if self._debug:
                print("lock:  rank {} got the lock".format(self._rank), flush=True)

        # We have the lock now!
        self._have_lock = True
        return

    def unlock(self):
        """
        Unlock and return.
        """
        # Do we even have the lock?
        if not self._have_lock:
            return

        if self._comm is not None:
            from mpi4py import MPI

            # lock the window
            if self._debug:
                print(
                    "unlock:  rank {}, instance {} locking shared window".format(
                        self._rank, self._tag
                    ),
                    flush=True,
                )
            self._win.Lock(self._root, MPI.LOCK_EXCLUSIVE)

            # Get a local copy of the buffer

            self._win.Get(
                [self._waiting, self._procs, self._mpitype],
                self._root,
                target=[0, self._procs, self._mpitype],
            )
            if self._debug:
                print(
                    "unlock:  rank {}, instance {} list = {}".format(
                        self._rank, self._tag, self._waiting
                    ),
                    flush=True,
                )

            # Get our wait number
            my_wait_val = self._waiting[self._rank]

            # Verify that no other processes have a number lower than ours.  If we
            # hold the lock, the wait numbers should only be increasing from here.
            # The only time these numbers reset to zero is when no process holds the
            # lock.
            invalid_indx = np.where(
                np.logical_and(self._waiting < my_wait_val, self._waiting > 0)
            )[0]
            if len(invalid_indx) > 0:
                print(
                    "rank {} has lock (wait number {}) and found ranks with lower wait numbers: {}".format(
                        self._rank, my_wait_val, self._waiting
                    ),
                    flush=True,
                )
                if self._comm is not None:
                    self._comm.Abort(1)

            # Find the next waiting process
            next_proc = np.where(self._waiting == my_wait_val + 1)[0]
            receiver = -1

            if len(next_proc) > 1:
                # This should never happen!
                print(
                    "rank {} has lock (wait number {}) and found multiple ranks next in line for token: {}".format(
                        self._rank, my_wait_val, self._waiting
                    ),
                    flush=True,
                )
                if self._comm is not None:
                    self._comm.Abort(1)
            elif len(next_proc) == 0:
                # There seems to be no processes waiting for the lock.  This implies
                # that there should also be no processes with even higher wait numbers.
                # Check this.
                invalid_indx = np.where(self._waiting > my_wait_val + 1)[0]
                if len(invalid_indx) > 0:
                    print(
                        "rank {} has lock (wait number {}) and found no rank next in line but other ranks with wait numbers: {}".format(
                            self._rank, my_wait_val, self._waiting
                        ),
                        flush=True,
                    )
                    if self._comm is not None:
                        self._comm.Abort(1)
            else:
                # There is one process waiting
                receiver = next_proc[0]

            # Update the waiting list
            my_wait = np.zeros((1,), dtype=self._dtype)

            self._win.Put(
                [my_wait, 1, self._mpitype],
                self._root,
                target=[self._rank, 1, self._mpitype],
            )
            if self._debug:
                print(
                    "unlock:  rank {}, instance {} reset wait to zero".format(
                        self._rank, self._tag
                    ),
                    flush=True,
                )

            # Flush

            self._win.Flush(self._root)

            # Release the window lock

            if self._debug:
                print(
                    "unlock:  rank {}, instance {} unlocking shared window".format(
                        self._rank, self._tag
                    ),
                    flush=True,
                )
            self._win.Unlock(self._root)

            # Send the token to the next process if one is waiting

            if receiver >= 0:
                if self._debug:
                    print(
                        "unlock:  rank {} sending the lock to {}".format(
                            self._rank, receiver
                        ),
                        flush=True,
                    )
                self._comm.Send(self._send_token, receiver, tag=self._tag)

            if self._debug:
                print("unlock:  rank {} sent the lock".format(self._rank), flush=True)

        self._have_lock = False
        return
