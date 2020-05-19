##
# Copyright (c) 2017-2020, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##

import numpy as np


def mpi_check_abort(comm, root, status, msg):
    """Check MPI return status.

    If the status is non-zero, print a message on the root process and abort.

    Args:
        comm (mpi4py.Comm): The communicator, or None.
        root (int): The root process.
        status (int): The MPI status.
        msg (str): The message to print in case of error.

    Returns:
        None

    """
    if comm is not None:
        from mpi4py import MPI

        failed = comm.allreduce(status, op=MPI.SUM)
        if failed > 0:
            if comm.rank == self._root:
                print(
                    "MPIShared: one or more processes failed: {}".format(msg),
                    flush=True,
                )
            comm.Abort()
    else:
        if status != 0:
            print("MPIShared: failed: {}".format(msg), flush=True)
            raise RuntimeError(msg)
    return


def mpi_data_type(comm, dt):
    """Helper function to return the byte size and MPI datatype.

    Args:
        comm (mpi4py.Comm): The communicator, or None.
        dt (np.dtype): The datatype.

    Returns:
        (tuple):  The (bytesize, MPI type) of the input dtype.

    """
    dtyp = np.dtype(dt)
    dsize = None
    mpitype = None
    if comm is None:
        dsize = dtyp.itemsize
    else:
        from mpi4py import MPI

        # We are actually using MPI, so we need to ensure that
        # our specified numpy dtype has a corresponding MPI datatype.
        status = 0
        try:
            # Technically this is an internal variable, but online
            # forum posts from the developers indicate this is stable
            # at least until a public interface is created.
            mpitype = MPI._typedict[dtyp.char]
        except:
            status = 1
        mpi_check_abort(comm, 0, status, "numpy to MPI type conversion")
        dsize = mpitype.Get_size()
    return (dsize, mpitype)
