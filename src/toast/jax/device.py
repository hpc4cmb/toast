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
    for slurm_var in ["SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"]:
        if slurm_var in os.environ:
            nb_devices = int(os.environ[slurm_var])
            return nb_devices
    # tries device lists
    for device_list in [
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HIP_VISIBLE_DEVICES",
    ]:
        if device_list in os.environ:
            nb_devices = len(os.environ[device_list].split(","))
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

    WARNING: no Jax function should be called before the call to `initialize` in this function.
    """
    log = Logger.get()
    # allocates memory as needed instead of preallocating a large block all at once
    # this is required to get multi-process / multi-node working properly
    # this needs to be set before calling any JAX function
    if "XLA_PYTHON_CLIENT_PREALLOCATE" in os.environ:
        log.warning(
            f"'XLA_PYTHON_CLIENT_PREALLOCATE' variable detected in environement. TOAST will overwrite it to 'false'."
        )
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # picks device
    if (
        (not disabled)
        and (node_procs > 1)
        and (os.environ.get("JAX_PLATFORM_NAME", "gpu") != "cpu")
    ):
        # gets id of device to be used by this process
        nb_devices = max(1, get_environement_nb_devices())
        device_id = node_rank % nb_devices
        # associate the device to this process
        try:
            jax.distributed.initialize(local_device_ids=[device_id])
        except ValueError:
            # We are not running within slurm or using openmpi.  Explicitly
            # use localhost
            log.warning("Cannot initialize jax with defaults, using localhost")
            jax.distributed.initialize(
                coordinator_address="127.0.0.1:12345",
                num_processes=node_procs,
                process_id=node_rank,
                local_device_ids=[device_id],
            )
        # displays information on the device picked
        local_device = jax_accel_get_device()
        log.debug(
            f"JAX rank {node_rank}/{node_procs} uses device number {device_id}/{nb_devices} ({local_device})"
        )
    else:
        local_device = jax_accel_get_device()
        log.debug(f"JAX rank {node_rank}/{node_procs} uses device {local_device}")
