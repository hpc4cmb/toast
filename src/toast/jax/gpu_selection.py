import jax

# list of GPUs / CPUs available
devices_available = jax.devices()
print(f"DEBUG: JAX devices:{devices_available}")

# the device used by JAX on this process
my_device = devices_available[0]
"""
    The device that JAX operators should be using
    Use the function `set_JAX_device` to set this value to something else than the first device
    Use like this in your JAX code: `jax.device_put(data, device=my_device)`
"""

# whether we are running on CPU
use_cpu = my_device.platform == "cpu"
"""
    Are we using the CPU?
"""


def set_JAX_device(process_id):
    """
    Sets `my_device` so that JAX functions can run on non-default devices.
    Inputs: `process_id` is the id of this process (from 0 to the number of process on the node).
    Outputs: `None`, this function modifies `my_device` in place.
    TODO call this function somewhere, if possible in a systematic way
    NOTE:
        this function is not needed if slurm can be asked to give a single GPU per process,
        then they will all see a single device and pick it
        this would make everything simpler (no need to call this function nor pick a device in JAX operators)
        `--gpus-per-task=1 --gpu-bind=single:1`
    """
    # gets id of device to be used by this process
    nb_devices = len(devices_available)
    device_id = process_id % nb_devices
    # sets the device
    global my_device
    my_device = devices_available[device_id]
    print(f"DEBUG: JAX uses device number {device_id} ({my_device})")
