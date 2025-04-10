# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Utilities useful in synthetic tests."""

import os

import numpy as np

from ...mpi import Comm


def create_outdir(mpicomm, topdir=None, subdir=None):
    """Create the top level output directory and per-test subdir.

    If `topdir` is not specified, a directory is created in the current
    working directory.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        topdir (str): the top-level directory to create.
        subdir (str): the sub directory for this test.

    Returns:
        str: full path to the subdir if specified, else the topdir.

    """
    if topdir is None:
        pwd = os.path.abspath(".")
        topdir = os.path.join(pwd, "toast_output")
    retdir = topdir
    if subdir is not None:
        retdir = os.path.join(topdir, subdir)
    if (mpicomm is None) or (mpicomm.rank == 0):
        os.makedirs(retdir, exist_ok=True)
    if mpicomm is not None:
        mpicomm.barrier()
    return retdir


def create_comm(mpicomm, single_group=False):
    """Create a toast communicator for testing.

    Use the specified MPI communicator to attempt to create 2 process groups.
    If less than 2 processes are used, create a single process group.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        single_group (bool):  If True, always use a single process group.

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
            if single_group:
                groupsize = worldsize
            else:
                groupsize = worldsize // 2
        toastcomm = Comm(world=mpicomm, groupsize=groupsize)
    return toastcomm


def close_data(data):
    """Make sure that data objects and comms are cleaned up."""
    cm = data.comm
    if cm.comm_world is not None:
        cm.comm_world.barrier()
    data.clear()
    del data
    cm.close()
    del cm


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
