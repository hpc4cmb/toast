from time import time
from enum import Enum
import jax

#------------------------------------------------------------------------------
# IMPLEMENTATION SELECTION

class ImplementationType(Enum):
    """Describes the various implementation kind"""
    COMPILED = 1
    NUMPY = 2
    JAX = 3

def select_implementation(f_compiled, f_numpy, f_jax, default_implementationType=ImplementationType.COMPILED):
    """
    Returns a transformed function that takes an additional argument to select the implementation
    """
    def f_switch(*args, implementationType=default_implementationType):
        if implementationType == ImplementationType.COMPILED:
            return f_compiled(*args)
        if implementationType == ImplementationType.NUMPY:
            return f_numpy(*args)
        if implementationType == ImplementationType.JAX:
            return f_jax(*args)
    # insures we preserve the name of at least one of the functions
    if default_implementationType == ImplementationType.COMPILED:
        f_switch.__name__ = f_compiled.__name__
    if default_implementationType == ImplementationType.NUMPY:
        f_switch.__name__ = f_numpy.__name__
    if default_implementationType == ImplementationType.JAX:
        f_switch.__name__ = f_jax.__name__
    return f_switch

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
#import os
#os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=64'

# TODO displays devices used by JAX
print(f"DEBUG: JAX devices:{jax.devices()}")

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
