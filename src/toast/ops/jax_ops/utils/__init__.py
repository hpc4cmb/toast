from time import time
from enum import Enum
import jax
import numpy

from ....timing import function_timer

from .mutableArray import MutableJaxArray
from .data_localization import assert_data_localization, dataMovementTracker

# ------------------------------------------------------------------------------
# GPU SELECTIONS

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


# ------------------------------------------------------------------------------
# IMPLEMENTATION SELECTION

from ....accelerator import use_accel_jax, use_accel_omp


class ImplementationType(Enum):
    """Describes the various implementation kind"""

    COMPILED = 1
    NUMPY = 2
    JAX = 3


# implementation used on cpu
default_cpu_implementationType = ImplementationType.COMPILED

# implementation used on gpu
default_gpu_implementationType = default_cpu_implementationType
if use_accel_jax:
    default_gpu_implementationType = ImplementationType.JAX
elif use_accel_omp:
    default_gpu_implementationType = ImplementationType.COMPILED


def function_of_implementationType(f_compiled, f_numpy, f_jax, implementationType):
    """returns one of the three input functions depending on the implementation type requested"""
    if implementationType == ImplementationType.JAX:
        return f_jax
    elif implementationType == ImplementationType.NUMPY:
        return f_numpy
    else:
        return f_compiled


def is_accel_function(f):
    """
    Returns true if one of the inputs of a function is called use_accel
    """
    # https://stackoverflow.com/a/40363565/6422174
    args_names = f.__code__.co_varnames[: f.__code__.co_argcount]
    return "use_accel" in args_names


def runtime_select_implementation(f_cpu, f_gpu):
    """
    Returns a function that is f_gpu when called with use_accel=True and f_cpu otherwise
    """
    # otherwise picks at runtime depending on the use_accel input
    def f(*args, **kwargs):
        use_accel = kwargs.get("use_accel", args[-1])
        if use_accel:
            return f_gpu(*args, **kwargs)
        else:
            return f_cpu(*args, **kwargs)

    return f


def select_implementation(
    f_compiled, f_numpy, f_jax, overide_implementationType=None, default_to_gpu=False
):
    """
    picks the implementation to use

    use default_gpu_implementationType when a function is called with use_accel=True and default_cpu_implementationType otherwise
    if overide_implementationType is set, use that implementation on cpu and gpu
    if default_to_gpu is set to true, use the gpu implementation on both cpu and gpu
    """
    # the implementations that will be used
    cpu_implementationType = (
        default_cpu_implementationType
        if (overide_implementationType is None)
        else overide_implementationType
    )
    gpu_implementationType = (
        default_gpu_implementationType
        if (overide_implementationType is None)
        else overide_implementationType
    )
    if default_to_gpu:
        cpu_implementationType = gpu_implementationType
    # the functions that will be used
    f_cpu = function_of_implementationType(
        f_compiled, f_numpy, f_jax, cpu_implementationType
    )
    f_gpu = function_of_implementationType(
        f_compiled, f_numpy, f_jax, gpu_implementationType
    )
    # wrap the functions in timers
    is_accel = is_accel_function(
        f_gpu
    )  # set this information aside now as the timer will destroy it
    f_cpu = function_timer(f_cpu)
    f_gpu = function_timer(f_gpu)
    # wrap the function to pick the implementation at runtime
    if is_accel:
        print(
            f"DEBUG: implementation picked in case of use_accel:{gpu_implementationType} ({f_gpu.__name__})"
        )
        return runtime_select_implementation(f_cpu, f_gpu)
    else:
        print(
            f"DEBUG: implementation picked (no use_accel input):{cpu_implementationType} ({f_cpu.__name__})"
        )
        return f_cpu


# ------------------------------------------------------------------------------
# TIMING


def get_compile_time(f):
    """
    Returns a transformed function that runs twice in order deduce and display its compile time
    """

    def f_timed(*args):
        # compile time + runtime
        start = time()
        f(*args)
        mid = time()
        # just runtime as compile time was cached
        result = f(*args)
        end = time()
        # computes the various times
        compile_plus_run_time = mid - start
        run_time = end - mid
        compile_time = max(0.0, compile_plus_run_time - run_time)
        # displays times and returns result
        print(f"DEBUG {f.__name__}: compile-time:{compile_time} run-time:{run_time}")
        return result

    # insures we preserve the name of the function
    f_timed.__name__ = f.__name__
    return f_timed
