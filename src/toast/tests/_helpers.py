# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

# from contextlib import contextmanager

from ..mpi import Comm

from ..data import Data

from .. import qarray as qa

from ..instrument import Focalplane, Telescope

from ..instrument_sim import fake_hexagon_focalplane

from ..observation import DetectorData, Observation


ZAXIS = np.array([0, 0, 1.0])


# These are helper routines for common operations used in the unit tests.


def create_outdir(mpicomm, subdir=None):
    """Create the top level output directory and per-test subdir.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        subdir (str): the sub directory for this test.

    Returns:
        str: full path to the test subdir if specified, else the top dir.

    """
    pwd = os.path.abspath(".")
    testdir = os.path.join(pwd, "toast_test_output")
    retdir = testdir
    if subdir is not None:
        retdir = os.path.join(testdir, subdir)
    if (mpicomm is None) or (mpicomm.rank == 0):
        if not os.path.isdir(testdir):
            os.mkdir(testdir)
        if not os.path.isdir(retdir):
            os.mkdir(retdir)
    if mpicomm is not None:
        mpicomm.barrier()
    return retdir


def create_comm(mpicomm):
    """Create a toast communicator.

    Use the specified MPI communicator to attempt to create 2 process groups.
    If less than 2 processes are used, create a single process group.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).

    Returns:
        toast.Comm: the 2-level toast communicator.

    """
    toastcomm = None
    if mpicomm is None:
        toastcomm = Comm(world=mpicomm)
    else:
        worldsize = mpicomm.size
        groupsize = 1
        if worldsize >= 2:
            groupsize = worldsize // 2
        toastcomm = Comm(world=mpicomm, groupsize=groupsize)
    return toastcomm


def create_telescope(group_size):
    """Create a fake telescope with at least one detector per process."""
    npix = 1
    ring = 1
    while 2 * npix < group_size:
        npix += 6 * ring
        ring += 1
    fp = fake_hexagon_focalplane(n_pix=npix)
    return Telescope("test", focalplane=fp)


def create_distdata(mpicomm, obs_per_group=1, samples=10):
    """Create a toast communicator and distributed data object.

    Use the specified MPI communicator to attempt to create 2 process groups,
    each with some observations.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        obs_per_group (int): the number of observations assigned to each group.
        samples (int): number of samples per observation.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    toastcomm = create_comm(mpicomm)
    data = Data(toastcomm)
    for obs in range(obs_per_group):
        oname = "test-{}-{}".format(toastcomm.group, obs)
        oid = obs_per_group * toastcomm.group + obs
        tele = create_telescope(toastcomm.group_size)
        # FIXME: for full testing we should set detranks as approximately the sqrt
        # of the grid size so that we test the row / col communicators.
        ob = Observation(
            tele, samples=samples, name=oname, UID=oid, comm=toastcomm.comm_group
        )
        data.obs.append(ob)
    return data


def uniform_chunks(samples, nchunk=100):
    """Divide some number of samples into chunks.

    This is often needed when constructing a TOD class, and usually we want
    the number of chunks to be larger than any number of processes we might
    be using for the unit tests.

    Args:
        samples (int): The number of samples.
        nchunk (int): The number of chunks to create.

    Returns:
        array: This list of chunk sizes.

    """
    chunksize = samples // nchunk
    chunks = np.ones(nchunk, dtype=np.int64)
    chunks *= chunksize
    remain = samples - (nchunk * chunksize)
    for r in range(remain):
        chunks[r] += 1
    return chunks


#
# @contextmanager
# def mpi_guard(comm=MPI.COMM_WORLD):
#     """Ensure that if one MPI process raises an exception, all of them do.
#
#     Args:
#         comm (mpi4py.MPI.Comm): The MPI communicator.
#
#     """
#     failed = 0
#     print(comm.rank, ": guard: enter", flush=True)
#     try:
#         print(comm.rank, ": guard: yield", flush=True)
#         yield
#     except:
#         print(comm.rank, ": guard: except", flush=True)
#         msg = "Exception on process {}:\n".format(comm.rank)
#         exc_type, exc_value, exc_traceback = sys.exc_info()
#         lines = traceback.format_exception(exc_type, exc_value,
#             exc_traceback)
#         msg += "\n".join(lines)
#         print(msg, flush=True)
#         failed = 1
#         print(comm.rank, ": guard: except done", flush=True)
#
#     print(comm.rank, ": guard: failcount reduce", flush=True)
#     failcount = comm.allreduce(failed, op=MPI.SUM)
#     if failcount > 0:
#         raise RuntimeError("One or more MPI processes raised an exception")
#     print(comm.rank, ": guard: done", flush=True)
#
#     return
