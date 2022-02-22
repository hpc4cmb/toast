# Copyright (c) 2015-2022 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import time

from ._libtoast import Logger

from ._libtoast import accel_enabled as omp_accel_enabled
from ._libtoast import accel_get_device as omp_accel_get_device
from ._libtoast import accel_present as omp_accel_present
from ._libtoast import accel_create as omp_accel_create
from ._libtoast import accel_update_device as omp_accel_update_device
from ._libtoast import accel_update_host as omp_accel_update_host
from ._libtoast import accel_delete as omp_accel_delete


enable_vals = ["1", "yes", "true"]

use_accel_omp = False
if "TOAST_GPU_OPENMP" in os.environ and os.environ["TOAST_GPU_OPENMP"] in enable_vals:
    if omp_accel_enabled():
        use_accel_omp = True
    else:
        log = Logger.get()
        msg = "TOAST_GPU_OPENMP enabled at runtime, but package was not built "
        msg += "with OpenMP target offload support.  Disabling."
        log.warning(msg)

use_accel_jax = False
if "TOAST_GPU_JAX" in os.environ and os.environ["TOAST_GPU_JAX"] in enable_vals:
    try:
        import jax
        import jax.numpy as jnp

        use_accel_jax = True
    except Exception:
        # There could be many possible exceptions...
        log = Logger.get()
        msg = "TOAST_GPU_JAX enabled at runtime, but jax is not "
        msg += "importable.  Disabling."
        log.warning(msg)

if use_accel_omp and use_accel_jax:
    log = Logger.get()
    msg = "OpenMP target offload and JAX cannot both be enabled at runtime."
    log.error(msg)
    raise RuntimeError(msg)


# Wrapper functions that work with either numpy arrays mapped to omp device memory
# or jax arrays.


def accel_enabled():
    """Returns True if any accelerator support is enabled."""
    return use_accel_jax or use_accel_omp


def accel_get_device():
    """Return the device ID assigned to this process."""
    if use_accel_omp:
        return omp_accel_get_device()
    elif use_accel_jax:
        # FIXME: what to return for jax?
        return 0
    else:
        raise RuntimeError("Accelerator support is not enabled")


def accel_present(data):
    """Check if data is present on the device.

    For OpenMP target offload, this checks if the input data has an entry in the
    global map of host to device pointers.  For jax, this tests if the input array
    is a jax array.

    Args:
        data (array):  The data to test.

    Returns:
        (bool):  True if the data is present on the device.

    """
    if use_accel_omp:
        return omp_accel_present(data)
    elif use_accel_jax:
        if isinstance(data, jnp.DeviceArray):
            return True
        else:
            return False
    else:
        raise RuntimeError("Accelerator support is not enabled")


def accel_create(data):
    """Create device buffers.

    Using the input data array, create a corresponding device array.  For OpenMP
    target offload, this allocates device memory and adds it to the global map
    of host to device pointers.  For jax arrays, this is a no-op, since those
    arrays are mapped and managed elsewhere.

    Args:
        data (array):  The host array.

    Returns:
        None

    """
    if use_accel_omp:
        omp_accel_create(data)
    elif use_accel_jax:
        pass
    else:
        raise RuntimeError("Accelerator support is not enabled")


def accel_update_device(data):
    """Update device buffers.

    Ensure that the input data is updated on the device.  For OpenMP target offload,
    this will do a host to device copy and return the input host object.  For jax,
    this will take the input (either a numpy or jax array) and return a jax array.

    Args:
        data (array):  The host array.

    Returns:
        (object):  Either the original input (for OpenMP) or a jax array.

    """
    if use_accel_omp:
        omp_accel_update_device(data)
        return data
    elif use_accel_jax:
        return jnp.DeviceArray(data)
    else:
        raise RuntimeError("Accelerator support is not enabled")


def accel_update_host(data):
    """Update host buffers.

    Ensure that the input data is updated on the host.  For OpenMP target offload,
    this will do a device to host copy and return the input (updated) host object.
    For jax, this will take the input (either a numpy or jax array) and return a
    numpy array.

    Args:
        data (array):  The host array.

    Returns:
        (object):  Either the updated input (for OpenMP) or a numpy array.

    """
    if use_accel_omp:
        omp_accel_update_host(data)
        return data
    elif use_accel_jax:
        if isinstance(data, jnp.DeviceArray):
            # Return a numpy array
            return data.copy()
        else:
            # Already on the host
            return data
    else:
        raise RuntimeError("Accelerator support is not enabled")


def accel_delete(data):
    """Delete device copy of the data.

    For OpenMP target offload, this deletes the device allocated memory and removes
    the host entry from the global memory map.  For jax, this is a no-op.

    Args:
        data (array):  The host array.

    Returns:
        None

    """
    if use_accel_omp:
        omp_accel_delete(data)
    elif use_accel_jax:
        pass
    else:
        raise RuntimeError("Accelerator support is not enabled")
