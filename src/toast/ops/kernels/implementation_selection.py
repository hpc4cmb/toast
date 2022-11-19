from enum import IntEnum
from functools import wraps
from ...utils import Logger
from ...timing import function_timer
from ...accelerator import use_accel_jax, use_accel_omp
from ...accelerator.data_localization import function_datamovementtracker


class ImplementationType(IntEnum):
    """Describes the various implementation kind"""

    DEFAULT = 0
    COMPILED = 1
    NUMPY = 2
    JAX = 3

def select_implementation_cpu(f_compiled, f_numpy, f_jax):
    """
    Builds a new function that will select an implementation at runtime depending on its 'implementation_type' input.
    Adds timers on all kernels.
    """
    # functions are timed by default
    f_compiled = function_timer(f_compiled)
    f_numpy = function_timer(f_numpy)
    f_jax = function_timer(f_jax)
    f_default_cpu = f_compiled  # sets the default on CPU
    cpu_functions = [f_default_cpu, f_compiled, f_numpy, f_jax]
    # pick a function at runtime
    @wraps(f_compiled)
    def f_wrapped(*args, implementation_type=ImplementationType.DEFAULT, **kwargs):
        # pick a function
        f = cpu_functions[implementation_type]
        # returns the result
        return f(*args, **kwargs)

    return f_wrapped


def select_implementation(f_compiled, f_numpy, f_jax):
    """
    Builds a new function that will select an implementation at runtime depending on its 'use_accel' and 'implementation_type' inputs.
    Adds timers on all kernels.
    Adds movement trackers on the GPU kernels.
    """
    # functions are timed by default
    f_compiled = function_timer(f_compiled)
    f_numpy = function_timer(f_numpy)
    f_jax = function_timer(f_jax)
    f_default_cpu = f_compiled  # sets the default on CPU
    cpu_functions = [f_default_cpu, f_compiled, f_numpy, f_jax]
    # we also track data movement on GPU functions
    f_compiled_gpu = function_datamovementtracker(f_compiled)
    f_numpy_gpu = function_datamovementtracker(f_numpy)
    f_jax_gpu = function_datamovementtracker(f_jax)
    f_default_gpu = (
        f_jax_gpu if use_accel_jax else f_compiled_gpu
    )  # picks a GPU default depending on flags
    gpu_functions = [f_default_gpu, f_compiled_gpu, f_numpy_gpu, f_jax_gpu]
    # pick a function at runtime
    @wraps(f_compiled)
    def f_wrapped(*args, implementation_type=ImplementationType.DEFAULT, **kwargs):
        # extracts the use_accel input
        use_accel = kwargs.get("use_accel", args[-1])
        # pick a function
        if use_accel:
            f = gpu_functions[implementation_type]
        else:
            f = cpu_functions[implementation_type]
        # returns the result
        return f(*args, **kwargs)

    return f_wrapped
