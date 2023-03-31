import math
import os
import jax
from .._libtoast import Logger


def jax_accel_get_device():
    """Returns the device currenlty used by JAX."""
    # gets local device if it has been designated
    local_device = jax.config.jax_default_device
    # otherwise gets the first device available
    if local_device is None:
        devices = jax.local_devices()
        if len(devices) > 0:
            local_device = devices[0]
    return local_device


def jax_accel_assign_device(node_procs, node_rank, disabled):
    """
    Assign processes to target devices.

    Args:
        node_procs (int): number of processes per node
        node_rank (int): rank of the current process, within the node
        disabled (bool): gpu computing is disabled

    Returns:
        None: the device is stored in JAX internal state

    WARNING: no Jax function should be called before this function.
    """
    log = Logger.get()
    # allocates memory as needed instead of preallocating a large block all at once
    # this is required to get multi-process / multi-node working properly
    # this needs to be set before calling any JAX function
    if 'XLA_PYTHON_CLIENT_PREALLOCATE' in os.environ:
        log.warning(f"'XLA_PYTHON_CLIENT_PREALLOCATE' variable detected in environement. TOAST will overwrite it to 'false'.")
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # picks device
    if (not disabled) and (node_procs > 1) and (os.environ.get('JAX_PLATFORM_NAME', 'gpu') != 'cpu'): 
        devices_available = jax.local_devices()
        # gets id of device to be used by this process
        nb_devices = len(devices_available)
        device_id = node_rank % nb_devices
        # sets device as default
        local_device = devices_available[device_id]
        jax.config.update("jax_default_device", local_device)
        # display information on the device picked
        log.debug(f"JAX rank {node_rank}/{node_procs} uses device number {device_id}/{nb_devices} ({local_device})")
    else:
        local_device = jax_accel_get_device()
        log.debug(f"JAX rank {node_rank}/{node_procs} uses device {local_device}")
