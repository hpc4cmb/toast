from time import time
from enum import Enum
import jax
import numpy

from .mutableArray import MutableJaxArray

#------------------------------------------------------------------------------
# INTERVAL INDEXING

def make_interval_mask(size, intervals):
    """
    Creates a mask of the given size
    the mask will select data in the union of the intervals
    NOTE: indexes presents in several intervals will be covered only once
    """
    mask = numpy.full(shape=size, fill_value=False)
    for interval in intervals:
        interval_start = interval['first']
        interval_end = interval['last']+1
        mask[interval_start:interval_end] = True
    return mask

def make_interval_indexes(intervals):
    """
    Creates an array of indexes that is quivalent to the concatenation of the given intervals
    """
    result = []
    for interval in intervals:
        interval_start = interval['first']
        interval_end = interval['last']+1
        interval_indexes = numpy.arange(start=interval_start, stop=interval_end)
        result.append(interval_indexes)
    return numpy.concatenate(result)

#------------------------------------------------------------------------------
# GPU SELECTIONS

# list of GPUs / CPUs available
devices_available = jax.devices()
print(f"DEBUG: JAX devices:{devices_available}")

# the device used by JAX on this process
# TODO this is currently not used
my_device = devices_available[0]
"""
    The device that JAX operators should be using
    Use the function `set_JAX_device` to set this value to something else than the first device
    Use like this in your JAX code: `jax.device_put(data, device=my_device)`
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

#------------------------------------------------------------------------------
# PMAP

# used so that CPUs can be detected as devices by pmap
# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=64'

def get_divisor_device_number(nb_devices, axis_size):
    """
    finds a size such that it is lower or equal to nb_device but divides axis_size
    """
    result = nb_devices
    while axis_size % result != 0:
        result -= 1
    return result

def reshape_for_pmap(data, nb_devices):
    """
    reshapes data so that its leading axis has one element per device
    """
    divisor_nb_device = get_divisor_device_number(nb_devices, data.shape[0])
    print(f"DEBUG: 'parallel_vmap' uses {divisor_nb_device} devices.")
    new_shape = (divisor_nb_device, -1) + tuple(data.shape[1:])
    return data.reshape(new_shape)

def reshape_from_pmap(data):
    """
    reshape data so that its leading axis is not one element per device anymore
    """
    new_shape = data.shape[0] * data.shape[1], *data.shape[2:]
    return data.reshape(new_shape)

def parallel_vmap(f, in_axes, out_axes=0):
    """
    Takes a function and applies pmap on top of vmap on it to parallelize it explicitely accros cpus/gpus
    reshaping inputs and output so that they are split across devices in a pmap friendly way
    NOTE: 
    - we suppose that there is a single output (not a tuple of outputs)
    - this function will use the largest number of devices that is a multiple of the batch dimenssion (maybe not all devices)
    TODO: does this function work and brings any improvements?
    """
    # reshapes data if it has an axis to be modified
    nb_devices = jax.device_count()
    def conditional_reshape(axes,input):
        if axes == 0: return reshape_for_pmap(input, nb_devices)
        if axes == None: return input
        else: raise Exception("'parallel_vmap' cannot be applied on axis other than 0!") 
    # function that reshapes data, call pmap then reshapes data back
    def f_pmap(*args):
        reshaped_inputs = (conditional_reshape(axes,input) for (axes,input) in zip(in_axes,args))
        f_vmap = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
        f_jitted = jax.jit(f_vmap) # jits here as jit(pmap) is often a bad thing
        reshaped_output = jax.pmap(f_jitted, in_axes=in_axes, out_axes=out_axes)(*reshaped_inputs)
        return reshape_from_pmap(reshaped_output)
    return f_pmap
