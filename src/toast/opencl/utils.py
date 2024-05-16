# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""OpenCL utilities.
"""
import os
import numpy as np

from ..utils import Logger

# FIXME: remove this once we no longer need a compiled extension.
from .._libtoast import (
    AlignedF32,
    AlignedF64,
    AlignedI8,
    AlignedI16,
    AlignedI32,
    AlignedI64,
    AlignedU8,
    AlignedU16,
    AlignedU32,
    AlignedU64,
)

# Check if pyopencl is importable

try:
    import pyopencl as cl

    have_opencl = True
except Exception:
    # There could be several possible exceptions...
    have_opencl = False
    log = Logger.get()
    msg = "pyopencl is not importable- disabling"
    log.debug(msg)


def find_source(calling_file, rel_path):
    """Locate an OpenCL source file relative to another file.

    Args:
        calling_file (str):  The __FILE__ of the caller.
        rel_path (str):  The relative path.

    Returns:
        (str):  The path to the file (or None)

    """
    path = os.path.join(calling_file, rel_path)
    apath = os.path.abspath(path)
    if not os.path.isfile(apath):
        msg = f"OpenCL source file '{apath}' does not exist"
        raise RuntimeError(msg)
    return apath


def aligned_to_dtype(aligned):
    """Return the dtype for an internal Aligned class.

    Args:
        aligned (class):  The Aligned class.

    Returns:
        (dtype):  The equivalent dtype.

    """
    log = Logger.get()
    if isinstance(aligned, AlignedI8):
        return np.dtype(np.int8)
    elif isinstance(aligned, AlignedU8):
        return np.dtype(np.uint8)
    elif isinstance(aligned, AlignedI16):
        return np.dtype(np.int16)
    elif isinstance(aligned, AlignedU16):
        return np.dtype(np.uint16)
    elif isinstance(aligned, AlignedI32):
        return np.dtype(np.int32)
    elif isinstance(aligned, AlignedU32):
        return np.dtype(np.uint32)
    elif isinstance(aligned, AlignedI64):
        return np.dtype(np.int64)
    elif isinstance(aligned, AlignedU64):
        return np.dtype(np.uint64)
    elif isinstance(aligned, AlignedF32):
        return np.dtype(np.float32)
    elif isinstance(aligned, AlignedF64):
        return np.dtype(np.float64)
    else:
        msg = f"Unsupported Aligned data class '{aligned}'"
        log.error(msg)
        raise ValueError(msg)


def get_kernel_deps(state, obs_name):
    """Extract kernel wait_for events for the current observation.

    Args:
        state (dict):  The state dictionary
        obs_name (str):  The observation name

    Returns:
        (list):  The list of events to wait on.

    """
    if obs_name is None:
        msg = "Observation name cannot be None"
        raise RuntimeError(msg)
    if state is None:
        # No dependencies
        # print(f"GET {obs_name}: state is None", flush=True)
        return list()
    if not isinstance(state, dict):
        msg = "kernel state should be a dictionary keyed on observation name"
        raise RuntimeError(msg)
    if obs_name not in state:
        # No dependencies for this observation
        # print(f"GET {obs_name}: obs_name not in state", flush=True)
        return list()
    # Return events
    return state[obs_name]


def clear_kernel_deps(state, obs_name):
    """Clear kernel events for a given observation.

    This should be done **after** the events are completed.

    Args:
        state (dict):  The state dictionary
        obs_name (str):  The observation name

    Returns:
        None

    """
    if obs_name is None:
        msg = "Observation name cannot be None"
        raise RuntimeError(msg)
    if state is None:
        # No dependencies
        return
    if not isinstance(state, dict):
        msg = "kernel state should be a dictionary keyed on observation name"
        raise RuntimeError(msg)
    if obs_name not in state:
        # No dependencies for this observation
        return
    # Clear
    state[obs_name].clear()


def replace_kernel_deps(state, obs_name, events):
    """Clear the events for a given observation and replace.

    The event list for the specified observation is created if needed.

    Args:
        state (dict):  The state dictionary
        obs_name (str):  The observation name
        events (Event, list):  pyopencl event or list of events.

    Returns:
        None

    """
    if obs_name is None:
        msg = "Observation name cannot be None"
        raise RuntimeError(msg)
    if state is None:
        msg = "State dictionary cannot be None"
        raise RuntimeError(msg)
    if not isinstance(state, dict):
        msg = "kernel state should be a dictionary keyed on observation name"
        raise RuntimeError(msg)
    if obs_name in state:
        state[obs_name].clear()
    else:
        state[obs_name] = list()
    if events is None:
        return
    if isinstance(events, list):
        state[obs_name].extend(events)
    else:
        state[obs_name].append(events)


def add_kernel_deps(state, obs_name, events):
    """Append event(s) to the current observation state.

    The event list for the specified observation is created if needed.

    Args:
        state (dict):  The state dictionary
        obs_name (str):  The observation name
        events (Event, list):  pyopencl event or list of events.

    Returns:
        None

    """
    if obs_name is None:
        msg = "Observation name cannot be None"
        raise RuntimeError(msg)
    if state is None:
        msg = "State dictionary cannot be None"
        raise RuntimeError(msg)
    if obs_name not in state:
        state[obs_name] = list()
    if events is None:
        return
    if isinstance(events, list):
        state[obs_name].extend(events)
    else:
        state[obs_name].append(events)
