# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import datetime
import json
import os
import re

import h5py
import numpy as np
from astropy import units as u

from ..instrument import Focalplane, GroundSite, SpaceSite, Telescope
from ..mpi import MPI, use_mpi
from ..observation import Observation
from ..timing import GlobalTimers, Timer, function_timer
from ..utils import (
    Environment,
    Logger,
    dtype_to_aligned,
    have_hdf5_parallel,
    import_from_name,
)
from ..weather import SimWeather


def check_dataset_buffer_size(msg, slices, dtype, parallel):
    """Check the buffer size that will be used for I/O.

    When using HDF5 parallel I/O, reading or writing to a dataset with
    a buffer size > 2GB will cause an error.  This function checks the
    buffer size and issues a warning to provide more user feedback.

    Args:
        msg (str):  Message to write
        slices (tuple):  The slices that will be used for I/O
        dtype (numpy.dtype):  The data type
        parallel (bool):  Whether parallel h5py is enabled.

    Returns:
        None

    """
    log = Logger.get()
    if not parallel:
        # No issues
        return
    nelem = 1
    for slc in slices:
        nelem *= slc.stop - slc.start
    nbytes = nelem * dtype.itemsize
    if nbytes >= 2147483647:
        wmsg = f"{msg}:  buffer size of {nbytes} bytes > 2^31 - 1. "
        wmsg += "  HDF5 parallel I/O will likely fail."
        log.warning(wmsg)


def hdf5_config(comm=None, force_serial=False):
    """Get information about how a process is involved in HDF I/O.

    Args:
        comm (MPI.Comm):  The optional MPI communicator.
        force_serial (bool):  If True, force serial access even if MPI is
            available.

    Returns:
        (tuple):  (use_parallel, participating, rank) Information about the
            configuration.  If use_parallel is True, it means that parallel
            HDF5 is available and should be used.  If participating is True,
            then this process is participating in the I/O.  Rank is the
            process rank within the participating processes.

    """
    parallel = have_hdf5_parallel()
    if force_serial:
        parallel = False
    rank = 0
    if comm is not None:
        rank = comm.rank
    participating = parallel or (rank == 0)
    return (parallel, participating, rank)


def hdf5_open(path, mode, comm=None, force_serial=False):
    """Open a file for reading or writing.

    This attempts to open the file with the mpio driver if available.  If
    not available or force_serial is True, then the file is opened on
    the rank zero process.

    Args:
        path (str):  The file path.
        mode (str):  The opening mode ("r", "w", etc).
        comm (MPI.Comm):  Optional MPI communicator.
        force_serial (bool):  If True, use serial HDF5 even if MPI is
            available.

    Returns:
        (h5py.File):  The opened file handle, or None if this process is
            not participating in I/O.

    """
    log = Logger.get()
    (parallel, participating, rank) = hdf5_config(comm=comm, force_serial=force_serial)
    hf = None
    if participating:
        if parallel:
            hf = h5py.File(path, mode, driver="mpio", comm=comm)
            if rank == 0:
                log.verbose(f"Opened file {path} in parallel")
        else:
            hf = h5py.File(path, mode)
            log.verbose(f"Opened file {path} serially")
    return hf


class H5File(object):
    """Wrapper class containing an open HDF5 file.

    If the file is opened in serial mode with an MPI communicator, then
    The open file handle will be None on processes other than rank 0.

    """

    def __init__(self, name, mode, comm=None, force_serial=False):
        self.handle = hdf5_open(name, mode, comm=comm, force_serial=force_serial)

    def close(self):
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
        self.handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
