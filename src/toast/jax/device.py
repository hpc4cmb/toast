import math
import os
import jax
from .._libtoast import Logger

def get_environement_nb_devices():
    """
    Returns the number of devices available on this node.
    (without calling a Jax function)

    FIXME: this function could be:
    - replaced by future mpi4py functionality,
    - moved to accel.py,
    - have its output passed to accel_get_device functions.
    """
    # tries slurm specific variables
    for slurm_var in ['SLURM_GPUS_ON_NODE', 'SLURM_GPUS_PER_NODE']:
        if slurm_var in os.environ:
            nb_devices = int(os.environ[slurm_var])
            return nb_devices
    # tries device lists
    for device_list in ['CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL', 'HIP_VISIBLE_DEVICES']:
        if device_list in os.environ:
            nb_devices = len(os.environ[device_list].split(','))
            return nb_devices
    # defaults to 1 in the absence of further information
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
        nb_devices = max(1, get_environement_nb_devices())
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
            f"JAX rank {node_rank}/{node_procs} uses {mem_percent}% of device {local_device} ({device_id+1}/{nb_devices})"
        )
