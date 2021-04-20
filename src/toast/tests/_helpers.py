# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

# from contextlib import contextmanager

from ..mpi import Comm

from ..dist import Data

from .. import qarray as qa


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


def create_distdata(mpicomm, obs_per_group=1):
    """Create a toast communicator and distributed data object.

    Use the specified MPI communicator to attempt to create 2 process groups,
    each with some observations.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        obs_per_group (int): the number of observations assigned to each group.

    Returns:
        toast.Data: the distributed data with named observations (but no TOD).

    """
    toastcomm = create_comm(mpicomm)
    data = Data(toastcomm)
    for obs in range(obs_per_group):
        ob = {}
        ob["name"] = "test-{}-{}".format(toastcomm.group, obs)
        ob["id"] = obs_per_group * toastcomm.group + obs
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


def boresight_focalplane(
    ndet,
    samplerate=1.0,
    epsilon=0.0,
    net=1.0,
    fmin=0.0,
    alpha=1.0,
    fknee=0.05,
    pairs=False,
):
    """Create a set of detectors at the boresight.

    This creates multiple detectors at the boresight, oriented in evenly
    spaced increments from zero to PI.

    Args:
        ndet (int): the number of detectors.
        pairs (bool):  add a polarization-orthogonal counterpart for every detector

    Returns:
        (tuple): names(list), quat(dict), fmin(dict), rate(dict), fknee(dict),
            alpha(dict), netd(dict)

    """

    if pairs:
        if ndet % 2 != 0:
            raise RuntimeError("Number of detectors must be even to make pairs")
        names = []
        pol = {}
        npixel = ndet // 2
        for pixel in range(npixel):
            for arm in "AB":
                det = "d{:02d}{}".format(pixel, arm)
                names.append(det)
                pol[det] = pixel * np.pi / npixel + (arm == "B") * np.pi / 2
    else:
        names = ["d{:02d}".format(x) for x in range(ndet)]
        pol = {"d{:02d}".format(x): (x * np.pi / ndet) for x in range(ndet)}

    quat = {}
    det_eps = {}
    det_fmin = {}
    det_rate = {}
    det_alpha = {}
    det_net = {}
    det_fknee = {}
    for idet, det in enumerate(names):
        quat[det] = qa.rotation(ZAXIS, pol[det])
        det_eps[det] = epsilon
        det_fmin[det] = fmin
        det_rate[det] = samplerate
        det_alpha[det] = alpha
        det_net[det] = net
        if np.isscalar(fknee):
            det_fknee[det] = fknee
        else:
            if len(fknee) != ndet:
                raise RuntimeError("length of knee frequencies must equal ndet")
            det_fknee[det] = fknee[idet]

    return names, quat, det_eps, det_rate, det_net, det_fmin, det_fknee, det_alpha


def create_weather(outfile):
    from astropy.table import Table
    import astropy.io.fits as af

    nstep = 101
    TQI = [np.linspace(0, 0.5, nstep) for x in range(24)]
    TQL = [np.linspace(0, 0.1, nstep) for x in range(24)]
    TQV = [np.linspace(1, 12, nstep) for x in range(24)]
    QV10M = [np.linspace(0.001, 0.007, nstep) for x in range(24)]
    PS = [np.linspace(58600, 59000, nstep) for x in range(24)]
    TS = [np.linspace(270, 280, nstep) for x in range(24)]
    T10M = [np.linspace(270, 280, nstep) for x in range(24)]
    U10M = [np.linspace(-2, 8, nstep) for x in range(24)]
    V10M = [np.linspace(-1, 3, nstep) for x in range(24)]

    hdus = af.HDUList([af.PrimaryHDU()])

    for mon in range(12):
        tab = Table(
            [TQI, TQL, TQV, QV10M, PS, TS, T10M, U10M, V10M],
            names=("TQI", "TQL", "TQV", "QV10M", "PS", "TS", "T10M", "U10M", "V10M"),
            meta={
                "PROBSTRT": 0.0,
                "PROBSTOP": 1.0,
                "PROBSTEP": 0.01,
                "NSTEP": nstep,
                "MONTH": mon,
            },
        )
        hdus.append(af.table_to_hdu(tab))

    hdus.writeto(outfile, overwrite=True)

    return


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
