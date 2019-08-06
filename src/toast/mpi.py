# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys
import itertools
import ctypes as ct

from .utils import Environment, Logger

env = Environment.get()

use_mpi = env.use_mpi()

MPI = None
MPI_Comm = None

if use_mpi:
    try:
        import mpi4py.MPI as MPI
    except ImportError:
        raise ImportError(
            "TOAST built with MPI + mpi4py support, but mpi4py "
            "not found at run time.  Is mpi4py currently in "
            "your python search path?"
        )
    try:
        if MPI._sizeof(MPI.Comm) == ct.sizeof(ct.c_int):
            MPI_Comm = ct.c_int
        else:
            MPI_Comm = ct.c_void_p
    except Exception as e:
        raise Exception(
            "Failed to set the portable MPI communicator datatype. MPI4py is "
            "probably too old. You need to have at least version 2.0. ({})".format(e)
        )


def comm_py2c(comm):
    if MPI_Comm is None:
        return None
    else:
        comm_ptr = MPI._addressof(comm)
        return MPI_Comm.from_address(comm_ptr)


def get_world():
    """Retrieve the default world communicator and its properties.

    If MPI is enabled, this returns MPI.COMM_WORLD and the process rank and number of
    processes.  If MPI is disabled, this returns None for the communicator, zero
    for the rank, and one for the number of processes.

    Returns:
        (tuple):  The communicator, number of processes, and rank.

    """
    rank = 0
    procs = 1
    world = None
    if use_mpi:
        world = MPI.COMM_WORLD
        rank = world.rank
        procs = world.size
    return world, procs, rank


class Comm(object):
    """Class which represents a two-level hierarchy of MPI communicators.

    A Comm object splits the full set of processes into groups of size
    "group".  If group_size does not divide evenly into the size of the given
    communicator, then those processes remain idle.

    A Comm object stores three MPI communicators:  The "world" communicator
    given here, which contains all processes to consider, a "group"
    communicator (one per group), and a "rank" communicator which contains the
    processes with the same group-rank across all groups.

    If MPI is not enabled, then all communicators are set to None.

    Args:
        world (mpi4py.MPI.Comm): the MPI communicator containing all processes.
        group (int): the size of each process group.

    """

    def __init__(self, world=None, groupsize=0):
        log = Logger.get()
        if world is None:
            if use_mpi:
                # Default is COMM_WORLD
                world = MPI.COMM_WORLD
            else:
                # MPI is disabled, leave world as None.
                pass
        else:
            if use_mpi:
                # We were passed a communicator to use. Check that it is
                # actually a communicator, otherwise fall back to COMM_WORLD.
                if not isinstance(world, MPI.Comm):
                    log.warning(
                        "Specified world communicator is not a valid "
                        "mpi4py.MPI.Comm object.  Using COMM_WORLD."
                    )
                    world = MPI.COMM_WORLD
            else:
                log.warning(
                    "World communicator specified even though "
                    "MPI is disabled.  Ignoring this constructor "
                    "argument."
                )
                world = None

        self._wcomm = world
        self._wrank = 0
        self._wsize = 1
        if self._wcomm is not None:
            self._wrank = self._wcomm.rank
            self._wsize = self._wcomm.size

        self._gsize = groupsize

        if (self._gsize < 0) or (self._gsize > self._wsize):
            log.warning(
                "Invalid groupsize ({}).  Should be between {} "
                "and {}.  Using single process group instead.".format(
                    groupsize, 0, self._wsize
                )
            )
            self._gsize = 0

        if self._gsize == 0:
            self._gsize = self._wsize

        self._ngroups = self._wsize // self._gsize

        if self._ngroups * self._gsize != self._wsize:
            msg = (
                "World communicator size ({}) is not evenly divisible "
                "by requested group size ({}).".format(self._wsize, self._gsize)
            )
            log.error(msg)
            raise RuntimeError(msg)

        self._group = self._wrank // self._gsize
        self._grank = self._wrank % self._gsize

        if self._ngroups == 1:
            # We just have one group with all processes.
            self._gcomm = self._wcomm
            if use_mpi:
                self._rcomm = MPI.COMM_SELF
            else:
                self._rcomm = None
        else:
            # We need to split the communicator.  This code is never executed
            # unless MPI is enabled and we have multiple groups.
            self._gcomm = self._wcomm.Split(self._group, self._grank)
            self._rcomm = self._wcomm.Split(self._grank, self._group)

    @property
    def world_size(self):
        """The size of the world communicator.
        """
        return self._wsize

    @property
    def world_rank(self):
        """The rank of this process in the world communicator.
        """
        return self._wrank

    @property
    def ngroups(self):
        """The number of process groups.
        """
        return self._ngroups

    @property
    def group(self):
        """The group containing this process.
        """
        return self._group

    @property
    def group_size(self):
        """The size of the group containing this process.
        """
        return self._gsize

    @property
    def group_rank(self):
        """The rank of this process in the group communicator.
        """
        return self._grank

    @property
    def comm_world(self):
        """The world communicator.
        """
        return self._wcomm

    @property
    def comm_group(self):
        """The communicator shared by processes within this group.
        """
        return self._gcomm

    @property
    def comm_rank(self):
        """The communicator shared by processes with the same group_rank.
        """
        return self._rcomm


#
# These general purpose MPI tools taken from:
#
#     https://github.com/tskisner/mpi_shmem
#
# Revision = 0c89cb0ec2ebfd32d47cda6b5bc87cfb8981ee9c
#

class MPILock(object):
    """
    Implement a MUTEX lock with MPI one-sided operations.

    The lock is created across the given communicator.  This uses an array
    of bytes (one per process) to track which processes have requested the
    lock.  When a given process releases the lock, it passes it to the next
    process in line.

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

        # Allocate the shared memory buffer.

        self._mpitype = None
        self._win = None
        self._have_lock = False
        self._waiting = None
        if self._rank == 0:
            self._waiting = np.zeros((self._procs,), dtype=np.uint8)

        if self._comm is not None:
            from mpi4py import MPI
            # Root allocates the buffer
            status = 0
            try:
                self._win = MPI.Win.Create(self._waiting, comm=self._comm)
            except:
                if self._debug:
                    print("rank {} win create raised exception".format(self._rank),
                          flush=True)
                status = 1
            self._checkabort(self._comm, status,
                             "shared memory allocation")

            self._comm.barrier()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
        return False

    def close(self):
        # Explicitly free the shared window
        if self._win is not None:
            self._win.Free()
            self._win = None
        return

    @property
    def comm(self):
        """
        The communicator.
        """
        return self._comm

    def _checkabort(self, comm, status, msg):
        from mpi4py import MPI
        failed = comm.allreduce(status, op=MPI.SUM)
        if failed > 0:
            if comm.rank == self._root:
                print("MPIShared: one or more processes failed: {}".format(
                    msg))
            comm.Abort()
        return

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
            waiting = np.zeros((self._procs,), dtype=np.uint8)
            lock = np.zeros((1,), dtype=np.uint8)
            lock[0] = 1

            # lock the window
            if self._debug:
                print("lock:  rank {}, instance {} locking shared window".format(
                    self._rank, self._tag), flush=True)
            self._win.Lock(self._root, MPI.LOCK_EXCLUSIVE)

            # add ourselves to the list of waiting ranks
            if self._debug:
                print("lock:  rank {}, instance {} putting rank".format(
                    self._rank, self._tag), flush=True)
            self._win.Put([lock, 1, MPI.UNSIGNED_CHAR],
                          self._root, target=self._rank)

            # get the full list of current processes waiting or running
            if self._debug:
                print("lock:  rank {}, instance {} getting waitlist".format(
                    self._rank, self._tag), flush=True)
            self._win.Get(
                [waiting, self._procs, MPI.UNSIGNED_CHAR], self._root)
            if self._debug:
                print("lock:  rank {}, instance {} list = {}".format(self._rank,
                                                                     self._tag, waiting), flush=True)

            self._win.Flush(self._root)

            # unlock the window
            if self._debug:
                print("lock:  rank {}, instance {} unlocking shared window".format(
                    self._rank, self._tag), flush=True)
            self._win.Unlock(self._root)

            # Go through the list of waiting processes.  If any one is
            # active or waiting, then wait for a signal that we can have
            # the lock.
            for p in range(self._procs):
                if (waiting[p] == 1) and (p != self._rank):
                    # we have to wait...
                    if self._debug:
                        print("lock:  rank {} waiting for the lock".format(self._rank),
                              flush=True)
                    self._comm.Recv(lock, source=MPI.ANY_SOURCE, tag=self._tag)
                    if self._debug:
                        print("lock:  rank {} got the lock".format(self._rank),
                              flush=True)
                    break

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
            waiting = np.zeros((self._procs,), dtype=np.uint8)
            lock = np.zeros((1,), dtype=np.uint8)

            # lock the window
            if self._debug:
                print("unlock:  rank {}, instance {} locking shared window"
                      .format(self._rank, self._tag), flush=True)
            self._win.Lock(self._root, MPI.LOCK_EXCLUSIVE)

            # remove ourselves to the list of waiting ranks
            if self._debug:
                print("unlock:  rank {}, instance {} putting rank".format(
                    self._rank, self._tag), flush=True)
            self._win.Put([lock, 1, MPI.UNSIGNED_CHAR], self._root,
                          target=self._rank)

            # get the full list of current processes waiting or running
            if self._debug:
                print("unlock:  rank {}, instance {} getting waitlist".format(
                    self._rank, self._tag), flush=True)
            self._win.Get([waiting, self._procs, MPI.UNSIGNED_CHAR],
                          self._root)
            if self._debug:
                print("unlock:  rank {}, instance {} list = {}"
                      .format(self._rank, self._tag, waiting), flush=True)

            self._win.Flush(self._root)

            # unlock the window
            if self._debug:
                print("unlock:  rank {}, instance {} unlocking shared window"
                      .format(self._rank, self._tag), flush=True)
            self._win.Unlock(self._root)

            # Go through the list of waiting processes.  Pass the lock
            # to the next process.
            next = self._rank + 1
            for p in range(self._procs):
                nextrank = next % self._procs
                if waiting[nextrank] == 1:
                    if self._debug:
                        print("unlock:  rank {} passing lock to {}"
                              .format(self._rank, nextrank), flush=True)
                    self._comm.Send(lock, nextrank, tag=self._tag)
                    self._have_lock = False
                    break
                next += 1
        else:
            self._have_lock = False

        return


class MPIShared(object):
    """
    Create a shared memory buffer that is replicated across nodes.

    For the given array dimensions and datatype, the original communicator
    is split into groups of processes that can share memory (i.e. that are
    on the same node).

    The values of the memory buffer can be set by one process at a time.
    When the set() method is called the data passed by the specified
    process is replicated to all nodes and then copied into the desired
    place in the shared memory buffer on each node.  This way the shared
    buffer on each node is identical.

    All processes across all nodes may do read-only access to their node-
    local copy of the buffer, simply by using the standard array indexing
    notation ("[]") on the object itself.

    Args:
        shape (tuple): the dimensions of the array.
        dtype (np.dtype): the data type of the array.
        comm (MPI.Comm): the full communicator to use.  This may span
            multiple nodes, and each node will have a copy.
    """

    def __init__(self, shape, dtype, comm):
        self._shape = shape
        self._dtype = dtype

        # Global communicator.

        self._comm = comm
        self._rank = 0
        self._procs = 1
        if self._comm is not None:
            self._rank = self._comm.rank
            self._procs = self._comm.size

        # Compute the flat-packed buffer size.

        self._n = 1
        for d in self._shape:
            self._n *= d

        # Split our communicator into groups on the same node.  Also
        # create an inter-node communicator between corresponding
        # processes on all nodes (for use in "setting" slices of the
        # buffer.

        self._nodecomm = None
        self._rankcomm = None
        self._noderank = 0
        self._nodeprocs = 1
        self._nodes = 1
        self._mynode = 0
        if self._comm is not None:
            import mpi4py.MPI as MPI
            self._nodecomm = self._comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
            self._noderank = self._nodecomm.rank
            self._nodeprocs = self._nodecomm.size
            self._nodes = self._procs // self._nodeprocs
            if self._nodes * self._nodeprocs < self._procs:
                self._nodes += 1
            self._mynode = self._rank // self._nodeprocs
            self._rankcomm = self._comm.Split(self._noderank, self._mynode)

        # Consider a corner case of the previous calculation.  Imagine that
        # the number of processes is not evenly divisible by the number of
        # processes per node.  In that case, when we later use the set()
        # method, the rank-wise communicator may not have a member on the
        # final node.  Here we compute the "highest" rank within a node which
        # is present on all nodes.  That sets the possible allowed processes
        # which may call the set() method.

        dist = self._disthelper(self._procs, self._nodes)
        self._maxsetrank = dist[-1][1] - 1

        # Divide up the total memory size among the processes on each
        # node.  For reasonable NUMA settings, this should spread the
        # allocated memory to locations across the node.

        # FIXME: the above statement works fine for allocating the window,
        # and it is also great in C/C++ where the pointer to the start of
        # the buffer is all you need.  In mpi4py, querying the rank-0 buffer
        # returns a buffer-interface-compatible object, not just a pointer.
        # And this "buffer object" has the size of just the rank-0 allocated
        # data.  SO, for now, disable this and have rank 0 allocate the whole
        # thing.  We should change this back once we figure out how to
        # take the raw pointer from rank zero and present it to numpy as the
        # the full buffer.

        # dist = self._disthelper(self._n, self._nodeprocs)
        # self._localoffset, self._nlocal = dist[self._noderank]
        if self._noderank == 0:
            self._localoffset = 0
            self._nlocal = self._n
        else:
            self._localoffset = 0
            self._nlocal = 0

        # Allocate the shared memory buffer and wrap it in a
        # numpy array.  If the communicator is None, just make
        # a normal numpy array.

        self._mpitype = None
        self._win = None

        self._buffer = None
        self._dbuf = None
        self._flat = None
        self._data = None

        if self._comm is None:
            dsize = self._dtype.itemsize
        else:
            import mpi4py.MPI as MPI
            # We are actually using MPI, so we need to ensure that
            # our specified numpy dtype has a corresponding MPI datatype.
            status = 0
            try:
                # Technically this is an internal variable, but online
                # forum posts from the developers indicate this is stable
                # at least until a public interface is created.
                self._mpitype = MPI._typedict[self._dtype.char]
            except:
                status = 1
            self._checkabort(self._comm, status,
                             "numpy to MPI type conversion")

            dsize = self._mpitype.Get_size()

        # Number of bytes in our buffer
        nbytes = self._nlocal * dsize

        self._buffer = None
        if self._comm is None:
            self._buffer = np.ndarray(shape=(nbytes,), dtype=np.dtype("B"),
                                      order="C")
        else:
            import mpi4py.MPI as MPI
            # Every process allocates a piece of the buffer.  The per-
            # process pieces are guaranteed to be contiguous.
            status = 0
            try:
                self._win = MPI.Win.Allocate_shared(nbytes, dsize,
                                                    comm=self._nodecomm)
            except:
                status = 1
            self._checkabort(self._nodecomm, status,
                             "shared memory allocation")

            # Every process looks up the memory address of rank zero's piece,
            # which is the start of the contiguous shared buffer.
            status = 0
            try:
                self._buffer, dsize = self._win.Shared_query(0)
            except:
                status = 1
            self._checkabort(self._nodecomm, status, "shared memory query")

        # Create a numpy array which acts as a "view" of the buffer.
        self._dbuf = np.array(self._buffer, dtype=np.dtype("B"), copy=False)
        self._flat = self._dbuf.view(self._dtype)
        self._data = self._flat.reshape(self._shape)

        # Initialize to zero.  Any of the processes could do this to the
        # whole buffer, but it is safe and easy for each process to just
        # initialize its local piece.

        # FIXME: change this back once every process is allocating a
        # piece of the buffer.
        # self._flat[self._localoffset:self._localoffset + self._nlocal] = 0
        if self._noderank == 0:
            self._flat[:] = 0

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
        return False

    def close(self):
        # Explicitly free the shared memory window.
        if self._win is not None:
            self._win.Free()
            self._win = None
        return

    @property
    def shape(self):
        """
        The tuple of dimensions of the shared array.
        """
        return self._shape

    @property
    def dtype(self):
        """
        The numpy datatype of the shared array.
        """
        return self._dtype

    @property
    def comm(self):
        """
        The full communicator.
        """
        return self._comm

    @property
    def nodecomm(self):
        """
        The node-local communicator.
        """
        return self._nodecomm

    def _disthelper(self, n, groups):
        dist = []
        for i in range(groups):
            myn = n // groups
            first = 0
            leftover = n % groups
            if i < leftover:
                myn += 1
                first = i * myn
            else:
                first = ((myn + 1) * leftover) + (myn * (i - leftover))
            dist.append((first, myn))
        return dist

    def _checkabort(self, comm, status, msg):
        import mpi4py.MPI as MPI
        failed = comm.allreduce(status, op=MPI.SUM)
        if failed > 0:
            if comm.rank == 0:
                print("MPIShared: one or more processes failed: {}".format(
                    msg))
                sys.stdout.flush()
            comm.Abort()
        return

    def set(self, data, offset, fromrank=0):
        """
        Set the values of a slice of the shared array.

        This call is collective across the full communicator, but only the
        data input from process "fromrank" is meaningful.  The offset
        specifies the starting element along each dimension when copying
        the data into the shared array.  Regardless of which node the
        "fromrank" process is on, the data will be replicated to the
        shared memory buffer on all nodes.

        Args:
            data (array): a numpy array with the same number of dimensions
                as the full array.
            offset (tuple): the starting offset along each dimension, which
                determines where the input data should be inserted into the
                shared array.
            fromrank (int): the process rank of the full communicator which
                is passing in the data.

        Returns:
            Nothing
        """
        # Explicit barrier here, to ensure that we don't try to update
        # data while other processes are reading.
        if self._comm is not None:
            self._comm.barrier()

        # First check that the dimensions of the data and the offset tuple
        # match the shape of the data.

        if self._rank == fromrank:
            if len(data.shape) != len(self._shape):
                if len(data.shape) != len(self._shape):
                    msg = "input data dimensions {} incompatible with "\
                        "buffer ({})".format(len(data.shape), len(self._shape))
                if self._comm is not None:
                    print(msg)
                    sys.stdout.flush()
                    self._comm.Abort()
                else:
                    raise RuntimeError(msg)
            if len(offset) != len(self._shape):
                msg = "input offset dimensions {} incompatible with "\
                    "buffer ({})".format(len(offset), len(self._shape))
                if self._comm is not None:
                    print(msg)
                    sys.stdout.flush()
                    self._comm.Abort()
                else:
                    raise RuntimeError(msg)
            if data.dtype != self._dtype:
                msg = "input data type ({}, {}) incompatible with "\
                    "buffer ({}, {})".format(data.dtype.str, data.dtype.num,
                                             self._dtype.str, self._dtype.num)
                if self._comm is not None:
                    print(msg)
                    sys.stdout.flush()
                    self._comm.Abort()
                else:
                    raise RuntimeError(msg)

        # The input data is coming from exactly one process on one node.
        # First, we broadcast the data from this process to the same node-rank
        # process on each of the nodes.

        if self._comm is not None:
            import mpi4py.MPI as MPI
            target_noderank = self._comm.bcast(self._noderank, root=fromrank)
            fromnode = self._comm.bcast(self._mynode, root=fromrank)

            # Verify that the node rank with the data actually has a member on
            # every node (see notes in the constructor).
            if target_noderank > self._maxsetrank:
                if self._rank == 0:
                    print("set() called with data from a node rank which does"
                          " not exist on all nodes")
                    self._comm.Abort()

            if self._noderank == target_noderank:
                # We are the lucky process on this node that gets to write
                # the data into shared memory!

                # Broadcast the offsets of the input slice
                copyoffset = None
                if self._mynode == fromnode:
                    copyoffset = offset
                copyoffset = self._rankcomm.bcast(copyoffset, root=fromnode)

                # Pre-allocate buffer, so that we can use the low-level
                # (and faster) Bcast method.
                datashape = None
                if self._mynode == fromnode:
                    datashape = data.shape
                datashape = self._rankcomm.bcast(datashape, root=fromnode)

                nodedata = None
                if self._mynode == fromnode:
                    nodedata = np.copy(data)
                else:
                    nodedata = np.zeros(datashape, dtype=self._dtype)

                # Broadcast the data buffer
                self._rankcomm.Bcast(nodedata, root=fromnode)

                # Now one process on every node has a copy of the data, and
                # can copy it into the shared memory buffer.

                dslice = []
                ndims = len(nodedata.shape)
                for d in range(ndims):
                    dslice.append(slice(copyoffset[d],
                                        copyoffset[d] + nodedata.shape[d], 1))
                slc = tuple(dslice)

                # Get a write-lock on the shared memory
                self._win.Lock(self._noderank, MPI.LOCK_EXCLUSIVE)

                # Copy data slice
                self._data[slc] = nodedata

                # Release the write-lock
                self._win.Unlock(self._noderank)

        else:
            # We are just copying to a numpy array...
            dslice = []
            ndims = len(data.shape)
            for d in range(ndims):
                dslice.append(slice(offset[d], offset[d] + data.shape[d], 1))
            slc = tuple(dslice)

            self._data[slc] = data

        # Explicit barrier here, to ensure that other processes do not try
        # reading data before the writing processes have finished.
        if self._comm is not None:
            self._comm.barrier()

        return

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        raise NotImplementedError("Setting individual array elements not"
                                  " supported.  Use the set() method instead.")
