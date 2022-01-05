from time import time
from enum import Enum

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
        print(f"DEBUG {f.func_name}: compile-time:{compile_time} run-time:{run_time}")
        return result
    return f_timed