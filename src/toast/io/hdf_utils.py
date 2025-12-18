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


def save_meta_object(parent, objname, obj):
    """Recursive function to save python metadata objects.

    This function attempts to make intelligent choices based on the object types:
        - Scalars are written as attributes to the parent group
        - Arrays are written as a dataset in the parent group
        - Dictionaries: a new group is created with objname and this function is
          called for each child key / value.
        - Lists / Tuples: a new group is created with objname and the original
          data type is recorded to a group attribute.  Next each item is passed
          to this function with a name "item_XXXX".

    Args:
        parent (h5py.Group):  The parent group (if this process is participating)
            else None.
        objname (str):  The name of the current object.
        obj (object):  A recognized object type (scalar, array, dict, list,
            set, tuple)

    Returns:
        None

    """
    log = Logger.get()

    def _type_to_str(obj):
        if isinstance(obj, dict):
            return "dict"
        elif isinstance(obj, list):
            return "list"
        elif isinstance(obj, tuple):
            return "tuple"
        else:
            msg = f"Unsupported container type '{type(obj)}'"
            raise ValueError(msg)

    if parent is None:
        # Not participating
        return

    if "type" not in parent.attrs:
        # This must be the root
        parent.attrs["type"] = "dict"

    if isinstance(obj, dict):
        child = parent.create_group(objname)
        child.attrs["type"] = _type_to_str(obj)
        for k, v in obj.items():
            save_meta_object(child, k, v)
    elif isinstance(obj, (list, tuple)):
        child = parent.create_group(objname)
        child.attrs["type"] = _type_to_str(obj)
        for indx, item in enumerate(obj):
            k = f"item_{indx:04d}"
            save_meta_object(child, k, item)
    elif isinstance(obj, u.Quantity):
        if isinstance(obj.value, np.ndarray):
            # Array quantity
            odata = parent.create_dataset(objname, data=obj.value)
            odata.attrs["units"] = obj.unit.to_string()
            del odata
        else:
            # Must be a scalar quantity
            parent.attrs[f"{objname}_value"] = obj.value
            parent.attrs[f"{objname}_units"] = obj.unit.to_string()
    elif isinstance(obj, np.ndarray):
        # Array
        arr = parent.create_dataset(objname, data=obj)
        del arr
    else:
        # This is a scalar or some kind of unknown object.  Try
        # to store it as an attribute and warn if something failed.
        try:
            parent.attrs[objname] = obj
        except (ValueError, TypeError) as e:
            msg = f"Failed to store metadata '{objname}' = '{v}' as an attribute ({e})."
            msg += " Ignoring."
            log.warn(msg)


def load_meta_object(parent):
    """Recursive function to load HDF5 metadata objects.

    This function recursively processes an HDF5 group and converts groups and
    datasets into python objects.

    Args:
        parent (h5py.Group):  The parent group (if this process is participating)
            else None.

    Returns:
        (object):  The populated python container

    """
    if "type" not in parent.attrs:
        raise RuntimeError("metadata group does not contain 'type' attribute")

    parsed = dict()
    parsed["type"] = parent.attrs["type"]

    # First process child groups / datasets
    for child_name in list(sorted(parent.keys())):
        if isinstance(parent[child_name], h5py.Group):
            # Descend
            child = parent[child_name]
            parsed[child_name] = load_meta_object(child)
            del child
        elif isinstance(parent[child_name], h5py.Dataset):
            child = parent[child_name]
            if "units" in child.attrs:
                # This is an array Quantity
                arr = u.Quantity(child, u.Unit(child.attrs["units"]), copy=True)
            else:
                # Plain numpy array
                arr = np.array(child, copy=True)
            parsed[child_name] = arr
            del child

    # Now process parent attributes
    units_pat = re.compile(r"(.*)_units")
    value_pat = re.compile(r"(.*)_value")
    for k, v in parent.attrs.items():
        if k == "type":
            continue
        if value_pat.match(k) is not None:
            # We will process this when matching units
            continue
        units_mat = units_pat.match(k)
        if units_mat is not None:
            # We have a quantity
            kname = units_mat.group(1)
            unit_str = v
            kval = parent.attrs[f"{kname}_value"]
            parsed[kname] = u.Quantity(kval, u.Unit(unit_str))
        else:
            # Simple scalar
            parsed[k] = v

    # If the parent container is a list or tuple, construct that now and sort the
    # children into the original order.
    ret = None
    ctype = parsed["type"]
    if ctype == "dict":
        del parsed["type"]
        return parsed
    elif ctype == "list":
        keys = list(sorted(parsed.keys()))
        ret = list()
        for k in keys:
            if k == "type":
                continue
            ret.append(parsed[k])
    elif ctype == "tuple":
        keys = list(sorted(parsed.keys()))
        ret = tuple()
        for k in keys:
            if k == "type":
                continue
            ret = ret + (parsed[k],)
    else:
        msg = f"Unsupported container format '{ctype}'"
        raise RuntimeError(msg)
    return ret
