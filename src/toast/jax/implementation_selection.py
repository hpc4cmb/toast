from enum import Enum
from ..timing import function_timer
from ..accelerator import use_accel_jax, use_accel_omp


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


def select_implementation(f_compiled, f_numpy, f_jax, overide_implementationType=None):
    """
    picks the implementation to use

    use default_gpu_implementationType when a function is called with use_accel=True and default_cpu_implementationType otherwise
    if overide_implementationType is set, use that implementation on cpu and gpu
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
    # the functions that will be used
    f_cpu = function_of_implementationType(
        f_compiled, f_numpy, f_jax, cpu_implementationType
    )
    f_gpu = function_of_implementationType(
        f_compiled, f_numpy, f_jax, gpu_implementationType
    )
    # wrap the functions in timers
    f_cpu = function_timer(f_cpu)
    f_gpu = function_timer(f_gpu)
    # wrap the function to pick the implementation at runtime
    print(
        f"DEBUG: implementation picked in case of use_accel:{gpu_implementationType} ({f_gpu.__name__})"
    )
    return runtime_select_implementation(f_cpu, f_gpu)
