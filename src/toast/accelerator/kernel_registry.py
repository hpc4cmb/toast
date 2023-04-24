# Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import inspect
from enum import IntEnum
from functools import wraps

from ..timing import function_timer
from .accel import accel_enabled
from .data_localization import function_datamovementtracker


class ImplementationType(IntEnum):
    """Describes the various implementation kinds"""

    DEFAULT = 0
    COMPILED = 1
    NUMPY = 2
    JAX = 3


registry = dict()


def kernel(impl, name=None):
    """Decorator which registers a kernel function.

    This will associate the function with a particular implementation of a kernel name.
    It checks the inputs and ensures that multiple functions are not registered for
    the same name / implementation.

    This returns the "dispatch" function for the kernel.  Since we have a decorator
    that takes some arguments, we must use a nested scheme.

    Args:
        impl (ImplementationType):  The implementation type
        name (str):  If not None, the kernel name to use for registration.  Otherwise
            the function name is used

    Returns:
        (function):  The dispatch function.

    """
    global registry

    if not isinstance(impl, ImplementationType):
        raise ValueError(
            "kernel decorator second argument should be ImplementationType"
        )

    def _kernel(f):
        nonlocal impl
        nonlocal name
        if name is None:
            name = f.__name__

        if name not in registry:
            # This is the first occurence of this kernel.  Set up the dictionary of
            # implementations and create a dispatch function.
            registry[name] = dict()

            def dispatch(
                *args, impl=ImplementationType.DEFAULT, use_accel=False, **kwargs
            ):
                return registry[name][impl](*args, use_accel=use_accel, **kwargs)

            registry[name]["dispatch"] = dispatch

        if impl in registry[name]:
            msg = f"Implementation {impl.name} for kernel {name} already exists."
            raise RuntimeError(msg)

        if impl == ImplementationType.DEFAULT:
            # When we register the default implementation, assign its docstring to
            # the dispatch function
            registry[name]["dispatch"].__doc__ = f.__doc__

        # Register the function with timer and data movement tracker
        if (
            (impl == ImplementationType.JAX) or (impl == ImplementationType.COMPILED)
        ) and accel_enabled():
            registry[name][impl] = function_timer(function_datamovementtracker(f))
        else:
            registry[name][impl] = function_timer(f)

        # Return the dispatch function in place of the original
        return registry[name]["dispatch"]

    return _kernel
