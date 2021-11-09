# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import numpy as np

import re
import datetime

from astropy import units as u

import h5py

import json

from ..utils import (
    Environment,
    Logger,
    import_from_name,
    dtype_to_aligned,
    have_hdf5_parallel,
)

from ..mpi import MPI

from ..timing import Timer, function_timer, GlobalTimers

from ..instrument import GroundSite, SpaceSite, Focalplane, Telescope

from ..weather import SimWeather

from ..observation import Observation


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
