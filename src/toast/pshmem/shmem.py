##
# Copyright (c) 2017-2020, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

import sys
import numpy as np

from .utils import mpi_check_abort, mpi_data_type


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
        # Copy the datatype in order to support arguments that are aliases,
        # like "numpy.float64".
        self._dtype = np.dtype(dtype)

        # Verify that our shape contains only integral values
        self._n = 1
        for d in shape:
            if not isinstance(d, (int, np.integer)):
                raise ValueError("input shape contains non-integer values")
            self._n *= d

        self._shape = tuple(shape)

        # Global communicator.

        self._comm = comm
        self._rank = 0
        self._procs = 1
        if self._comm is not None:
            self._rank = self._comm.rank
            self._procs = self._comm.size

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
            from mpi4py import MPI

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

        # Compute the data sizes
        self._dsize, self._mpitype = mpi_data_type(self._comm, self._dtype)

        # Allocate the shared memory buffer and wrap it in a
        # numpy array.  If the communicator is None, just make
        # a normal numpy array.

        self._win = None
        self._buffer = None
        self._dbuf = None
        self._flat = None
        self.data = None

        # Number of bytes in our buffer
        nbytes = self._nlocal * self._dsize

        self._win = None
        self._buffer = None
        if self._comm is None:
            self._buffer = np.ndarray(shape=(nbytes,), dtype=np.dtype("B"), order="C")
        else:
            import mpi4py.MPI as MPI

            # Every process allocates a piece of the buffer.  The per-
            # process pieces are guaranteed to be contiguous.
            status = 0
            try:
                self._win = MPI.Win.Allocate_shared(
                    nbytes,
                    disp_unit=self._dsize,
                    info=MPI.INFO_NULL,
                    comm=self._nodecomm,
                )
            except:
                status = 1
            mpi_check_abort(self._nodecomm, 0, status, "shared memory allocation")

            # Every process looks up the memory address of rank zero's piece,
            # which is the start of the contiguous shared buffer.
            status = 0
            try:
                self._buffer, dsize = self._win.Shared_query(0)
            except:
                status = 1
            mpi_check_abort(self._nodecomm, 0, status, "shared memory query")

        # Create a numpy array which acts as a "view" of the buffer.
        self._dbuf = np.array(self._buffer, dtype=np.dtype("B"), copy=False)
        self._flat = self._dbuf.view(self._dtype)
        self.data = self._flat.reshape(self._shape)

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
        if hasattr(self, "_win") and (self._win is not None):
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
                    msg = (
                        "input data dimensions {} incompatible with "
                        "buffer ({})".format(len(data.shape), len(self._shape))
                    )
                if self._comm is not None:
                    print(msg, flush=True)
                    self._comm.Abort()
                else:
                    raise RuntimeError(msg)
            if len(offset) != len(self._shape):
                msg = (
                    "input offset dimensions {} incompatible with "
                    "buffer ({})".format(len(offset), len(self._shape))
                )
                if self._comm is not None:
                    print(msg, flush=True)
                    self._comm.Abort()
                else:
                    raise RuntimeError(msg)
            if data.dtype != self._dtype:
                msg = (
                    "input data type ({}, {}) incompatible with "
                    "buffer ({}, {})".format(
                        data.dtype.str, data.dtype.num, self._dtype.str, self._dtype.num
                    )
                )
                if self._comm is not None:
                    print(msg, flush=True)
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
                    print(
                        "set() called with data from a node rank which does"
                        " not exist on all nodes",
                        flush=True,
                    )
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
                    dslice.append(
                        slice(copyoffset[d], copyoffset[d] + nodedata.shape[d], 1)
                    )
                slc = tuple(dslice)

                # Get a write-lock on the shared memory
                self._win.Lock(self._noderank, MPI.LOCK_EXCLUSIVE)

                # Copy data slice
                self.data[slc] = nodedata

                # Release the write-lock
                self._win.Unlock(self._noderank)

        else:
            # We are just copying to a numpy array...
            dslice = []
            ndims = len(data.shape)
            for d in range(ndims):
                dslice.append(slice(offset[d], offset[d] + data.shape[d], 1))
            slc = tuple(dslice)

            self.data[slc] = data

        # Explicit barrier here, to ensure that other processes do not try
        # reading data before the writing processes have finished.
        if self._comm is not None:
            self._comm.barrier()

        return

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        raise NotImplementedError(
            "Setting individual array elements not"
            " supported.  Use the set() method instead."
        )
