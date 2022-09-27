from time import time
from enum import Enum
import jax
import numpy

from ....timing import function_timer

from .mutableArray import MutableJaxArray
from .data_localization import assert_data_localization, dataMovementTracker

#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# IMPLEMENTATION SELECTION

from ....accelerator import use_accel_jax, use_accel_omp

class ImplementationType(Enum):
    """Describes the various implementation kind"""
    COMPILED = 1
    NUMPY = 2
    JAX = 3

default_implementationType=ImplementationType.JAX
"""
Implementation to be used in the absence of user specified information.
"""

def select_implementation(f_compiled, f_numpy, f_jax, 
                          overide_implementationType=None):
    """
    Returns one of the function depending on the settings.
    Set `default_implementationType` if you want one implementation in particular.
    `default_implementationType` defines the implementation that will be used in the absence of further information.
    """
    # picks the implementation to be used
    implementationType = overide_implementationType
    if implementationType is None:
        if use_accel_jax:
            implementationType = ImplementationType.JAX
        elif use_accel_omp:
            implementationType = ImplementationType.COMPILED
        else:
            implementationType = default_implementationType
    # returns the corresponding function
    if implementationType == ImplementationType.COMPILED:
        f = f_compiled
    elif implementationType == ImplementationType.NUMPY:
        f = f_numpy
    else: #implementationType == ImplementationType.JAX:
        f = f_jax
    print(f"DEBUG: implementation picked:{implementationType} ({f.__name__})")
    return f

def is_accel_function(f):
    """
    Returns true if one of the inputs of a funciton is called use_accel
    """
    # https://stackoverflow.com/a/40363565/6422174
    args_names = f.__code__.co_varnames[:f.__code__.co_argcount]
    return ('use_accel' in args_names)

def runtime_select_implementation(f_accel, f_host):
    """
    decides at runtime on whether to use the gpu version of a function or its cpu variant
    depending on the use_accel input
    """
    def f(*args, use_accel=False):
        if use_accel:
            return f_accel(*args)
        else:
            return f_host(*args)
    return f

def select_implementation(f_compiled, f_numpy, f_jax, 
                          overide_implementationType=None):
    """
    Returns a function that will default to f_compiled for all functions 
    except the ones having `use_accel` as one of their arguments 
    In which case, uses one of the function depending on the settings.
    Set `default_implementationType` if you want one implementation in particular.
    `default_implementationType` defines the implementation that will be used in the absence of further information.
    """
    # defaults to f_compiled if the function has no accel input
    if not is_accel_function(f_jax):
        return f_compiled
    # picks the implementation type to be used on GPU
    implementationType = overide_implementationType
    if implementationType is None:
        if use_accel_jax:
            implementationType = ImplementationType.JAX
        elif use_accel_omp:
            implementationType = ImplementationType.COMPILED
        else:
            implementationType = default_implementationType
    # gets the corresponding function
    if implementationType == ImplementationType.COMPILED:
        f_accel = f_compiled
    elif implementationType == ImplementationType.NUMPY:
        f_accel = f_numpy
    else: #implementationType == ImplementationType.JAX:
        f_accel = f_jax
    # wrap the funciton to use the compiled version on cpu
    f = runtime_select_implementation(f_accel, f_compiled)
    # wraps the function in a function timer
    f = function_timer(f)
    print(f"DEBUG: implementation picked in case of use_accel:{implementationType} ({f_accel.__name__})")
    return f

def select_implementation(f_compiled, f_numpy, f_jax, 
                          overide_implementationType=None):
    # defaults to f_compiled if the function has no accel input
    if not is_accel_function(f_jax):
        return f_compiled
    # picks the implementation type to be used on GPU
    implementationType = overide_implementationType
    if implementationType is None:
        if use_accel_jax:
            implementationType = ImplementationType.JAX
        elif use_accel_omp:
            implementationType = ImplementationType.COMPILED
        else:
            implementationType = default_implementationType
    # gets the corresponding function
    if implementationType == ImplementationType.COMPILED:
        f_accel = f_compiled
    elif implementationType == ImplementationType.NUMPY:
        f_accel = f_numpy
    else: #implementationType == ImplementationType.JAX:
        f_accel = f_jax
    # wraps the function in a fucntion timer
    f = function_timer(f_accel)
    print(f"DEBUG: implementation picked in case of use_accel:{implementationType} ({f_accel.__name__})")
    return f

#------------------------------------------------------------------------------
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
        compile_time = max(0., compile_plus_run_time - run_time)
        # displays times and returns result
        print(f"DEBUG {f.__name__}: compile-time:{compile_time} run-time:{run_time}")
        return result
    # insures we preserve the name of the function
    f_timed.__name__ = f.__name__
    return f_timed
