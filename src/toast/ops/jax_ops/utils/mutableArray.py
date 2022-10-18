import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from pshmem import MPIShared
from ....utils import AlignedI64, AlignedF64


class MutableJaxArray:
    """
    This class encapsulate a jax array to give the illusion of mutability
    simplifying integration within toast
    It is NOT designed for computation but, rather, as a container
    """

    data: jnp.DeviceArray
    shape: Tuple
    dtype: np.dtype
    nbytes: np.int64

    def __init__(self, cpu_data, gpu_data=None):
        """
        encapsulate an array as a jax array
        you can pass `gpu_data` (properly shaped) if you want to avoid a data transfert
        """
        # gets the data into jax
        self.cpu_data = cpu_data
        if gpu_data is None:
            data = MutableJaxArray.to_array(cpu_data)
            self.data = jax.device_put(data)
        else:
            self.data = gpu_data

        # gets basic information on the data
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        self.nbytes = self.data.nbytes

    #def zeros(shape, dtype=None):
    #    """creates an array of zeros"""
    #    data = jnp.zeros(shape=shape, dtype=dtype)
    #    return MutableJaxArray(data)

    def to_array(input):
        """
        Casts all array types used in TOAST to JAX-compatible array types
        (either a numpy array or a jax array)
        """
        if isinstance(input, MutableJaxArray) or isinstance(input, MPIShared):
            # those types need to be cast into their inner data
            return input.data
        elif isinstance(input, AlignedI64) or isinstance(input, AlignedF64):
            # NOTE: get inner array field, raw numpy conversion is very expensive
            return input.array()
        elif isinstance(input, np.ndarray) or isinstance(input, jax.numpy.ndarray):
            # those types can be feed to JAX raw with no error or performance problems
            return input
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
        self.cpu_data[:] = self.data
        return self.cpu_data

    def __setitem__(self, key, value):
        """
        updates the inner array in place
        """
        if key == slice(None):
            # optimises the [:] case
            self.data = value
        else:
            self.data = self.data.at[key].set(value)

    def __getitem__(self, key):
        """
        access the inner array
        """
        return self.data[key]

    def reshape(self, shape):
        """
        produces a new mutable array with a different shape
        the array has a copy of the gpu data (changes will not be propagated to the original)
        and a reshape of the cpu data (change might be propagated to the original, depending on numpy's behaviour)
        """
        reshaped_cpu_data = np.reshape(self.cpu_data, newshape=shape)
        reshaped_gpu_data = jnp.reshape(self.data, newshape=shape)
        return MutableJaxArray(reshaped_cpu_data, reshaped_gpu_data)

    def __str__(self):
        """
        returns a string representation of the content of the array
        """
        return self.data.__str__()

    def __eq__(self, other):
        raise RuntimeError(
            "MutableJaxArray: tried an equality test on a MutableJaxArray. This container is not designed for computations, you likely have a data movement bug somewhere in your program."
        )
