import math
import os
import jax
from .._libtoast import Logger

def slurm_nb_devices():
    """
    Returns the number of devices available on this node.
    (without calling a Jax function)
    """
    if 'SLURM_GPUS_ON_NODE' in os.environ:
        return int(os.environ['SLURM_GPUS_ON_NODE'])
    elif 'SLURM_GPUS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_GPUS_PER_NODE'])
    else:
        return 1

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

    WARNING: no Jax function should be called before the call to `initialize` inside this function.
    """
    if (not disabled) and (node_procs > 1): 
        # gets id of device to be used by this process
        nb_devices = slurm_nb_devices()
        device_id = node_rank % nb_devices
        process_per_device = math.ceil(node_procs / nb_devices)
        # sets the size of the preallocated memory pool
        # (this will only work if Jax has not be initialized yet)
        # we work with integers first to insure rounding down by a percent
        default_mem_fraction = float(os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 0.9))
        default_mem_percent = int(100 * default_mem_fraction)
        mem_percent = min(default_mem_percent, default_mem_percent // process_per_device)
        mem_fraction = mem_percent / 100
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f"{mem_fraction:.2f}"
        # associate the device to this process
        jax.distributed.initialize(local_device_ids=[device_id])
        # displays information on the device picked
        local_device = jax_accel_get_device()
        log = Logger.get()
        log.debug(
            f"JAX rank {node_rank}/{node_procs} uses device number {device_id}/{nb_devices} ({local_device} mem_fraction:{mem_fraction:.2f})"
        )
