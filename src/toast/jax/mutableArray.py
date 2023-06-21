from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from pshmem import MPIShared

from ..timing import function_timer
from ..utils import AlignedF64, AlignedI64, Logger

# ------------------------------------------------------------------------------
# SET ITEM


def convert_to_tuple(obj):
    """
    Converts all slices in a key, used to index an array, into tuples.
    """
    if isinstance(obj, slice):
        return ("slice", obj.start, obj.stop, obj.step)
    elif isinstance(obj, tuple):
        return tuple(convert_to_tuple(x) for x in obj)
    else:
        return obj


def convert_from_tuple(obj):
    """
    Convert slices, in a key used to index an array, back into slices.
    """
    if isinstance(obj, tuple):
        if isinstance(obj[0], str) and (obj[0] == "slice"):
            return slice(obj[1], obj[2], obj[3])
        else:
            return tuple(convert_from_tuple(x) for x in obj)
    else:
        return obj


def _setitem(data, key, value):
    """
    data[key] = value

    NOTE: key needs to be preprocessed with `convert_to_tuple` in order to make it hasheable.
    """
    # debugging information
    log = Logger.get()
    log.debug("MutableJaxArray.__setitem__: jit-compiling.")

    key = convert_from_tuple(key)
    return data.at[key].set(value)


# compiles the function, recycling the memory
_setitem_jitted = jax.jit(_setitem, donate_argnums=0, static_argnames="key")

# ------------------------------------------------------------------------------
# RESHAPE


def _reshape(data, newshape):
    """reshapes the data"""
    # debugging information
    log = Logger.get()
    log.debug("MutableJaxArray.reshape: jit-compiling.")

    return jnp.reshape(data, newshape=newshape)


# compiles the function, recycling the memory
_reshape_jitted = jax.jit(_reshape, donate_argnums=0, static_argnames="newshape")

# ------------------------------------------------------------------------------
# ZERO OUT


def _zero_out(data, output_shape=None):
    """Fills the data with zero."""
    # debugging information
    log = Logger.get()
    log.debug("MutableJaxArray.zero_out: jit-compiling.")

    if output_shape is None:
        return jnp.zeros_like(data)
    else:
        return jnp.zeros(shape=output_shape, dtype=data.dtype)


# compiles the function, recycling the memory
_zero_out_jitted = jax.jit(_zero_out, donate_argnums=0, static_argnames="output_shape")

# ------------------------------------------------------------------------------
# MUTABLE ARRAY


class MutableJaxArray:
    """
    This class encapsulate a jax array to give the illusion of mutability
    simplifying integration within toast
    It is NOT designed for computation but, rather, as a container
    """

    host_data: np.array
    data: jnp.DeviceArray
    shape: Tuple
    size: int
    dtype: np.dtype
    nbytes: np.int64

    def __init__(self, cpu_data, gpu_data=None):
        """
        encapsulate an array as a jax array
        you can pass `gpu_data` (properly shaped) if you want to avoid a data transfert
        """
        # stores cpu and gpu data
        if isinstance(cpu_data, MutableJaxArray):
            self.host_data = cpu_data.host_data
            self.data = cpu_data.data
        else:
            self.host_data = cpu_data
            if gpu_data is None:
                data = MutableJaxArray.to_array(cpu_data)
                self.data = jax.device_put(data)

        # use preset gpu_data if available
        if gpu_data is not None:
            self.data = gpu_data

        # gets basic information on the data
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        self.nbytes = self.data.nbytes

    @staticmethod
    def to_array(input):
        """
        Casts all array types used in TOAST to JAX-compatible array types
        (either a numpy array or a jax array)
        """
        if isinstance(input, MPIShared):
            # this type needs to be cast to its inner type
            # NOTE: the inner type might be a mutable array or a normal array
            input = input.data
        if isinstance(input, MutableJaxArray):
            # those types need to be cast into their inner data
            return input.data
        elif hasattr(input, "array"):
            # This is a wrapped C++ aligned memory buffer.  Get an array reference.
            return input.array()
        elif isinstance(input, np.ndarray) or isinstance(input, jax.numpy.ndarray):
            # those types can be feed to JAX raw with no error or performance problems
            return input
        elif isinstance(input, memoryview):
            # viewed as a numpy array first
            # FIXME: there might be a more efficient way to deal with this input type
            return np.asarray(input)
        else:
            # errors-out on other datatypes
            # so that we can make sure we are using the most efficient convertion available
            raise RuntimeError(
                f"Passed a {type(input)} to MutableJaxArray.to_array. Please find the best way to convert it to a Numpy array and update the function."
            )

    def to_host(self):
        """
        updates the original host container and returns it
        NOTE: we purposefully do not overload __array__ to avoid accidental conversions
        """
        try:
            self.host_data.array()[:] = self.data
        except AttributeError:
            # Not a wrapped C++ aligned type
            self.host_data[:] = self.data
        return self.host_data

    def __setitem__(self, key, value):
        """
        updates the inner array in place
        """
        if (key == slice(None)) and isinstance(value, jax.numpy.ndarray):
            # optimises the [:] case
            self.data = value
        else:
            # uses a compiled function that will recycle memory
            # instead of self.data = self.data.at[key].set(value)
            hasheable_key = convert_to_tuple(key)
            self.data = _setitem_jitted(self.data, hasheable_key, value)

    def __getitem__(self, key):
        """
        access the inner array
        """
        return self.data[key]

    @function_timer
    def reshape(self, shape):
        """
        produces a new mutable array with a different shape
        the array has a copy of the gpu data (changes will not be propagated to the original)
        and a reshape of the cpu data (change might be propagated to the original, depending on numpy's behaviour)
        """
        reshaped_cpu_data = np.reshape(self.host_data, newshape=shape)
        reshaped_gpu_data = _reshape_jitted(self.data, newshape=shape)
        return MutableJaxArray(reshaped_cpu_data, reshaped_gpu_data)

    def zero_out(self):
        """
        fills GPU data with zeros
        """
        self.data = _zero_out_jitted(self.data)

    def __str__(self):
        """
        returns a string representation of the content of the array
        """
        return self.data.__str__()

    def __eq__(self, other):
        raise RuntimeError(
            "MutableJaxArray: tried an equality test on a MutableJaxArray. This container is not designed for computations, you likely have a data movement bug somewhere in your program."
        )

    def __len__(self):
        """returns the length of the inner array"""
        return len(self.data)
