import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from pshmem import MPIShared

class MutableJaxArray():
    """
    This class encapsulate a jax array to give the illusion of mutability
    simplifying integration within toast
    It is NOT designed for computation but, rather, as a container
    """
    data: jnp.DeviceArray
    shape: Tuple
    dtype: np.dtype
    nbytes: np.int64

    def __init__(self, data):
        """encapsulate an array as a jax array"""
        # gets the data into jax
        if isinstance(data, np.ndarray):
            # converts to jax while insuring we send data to GPU
            # NOTE: device_put is faster than jnp.array, especially on CPU
            self.data = jax.device_put(data)
        elif isinstance(data, jax.numpy.ndarray):
            # already a jax array, does nothing to avoid useless copying
            self.data = data
        elif isinstance(data, MPIShared):
            # not numpy compatible enough for device_put
            # NOTE: get inner numpy field, raw numpy conversion is very expensive
            data = data.data
            self.data = jax.device_put(data)
        else:
            # errors-out on non numpy arrays 
            # so that we can make sure we are using the most efficient convertion available
            raise RuntimeError(f"Passed a {type(data)} to MutableJaxArray. Please find the best way to convert it to a Numpy array and update the function.")

        # gets basic information on the data
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        self.nbytes = self.data.nbytes

    def zeros(shape, dtype=None):
        """creates an array of zeros"""
        data = jnp.zeros(shape=shape, dtype=dtype)
        return MutableJaxArray(data)

    def to_array(input):
        """
        casts array types and, in particular, MutableJaxArray and MPIShared, to JAX-compatible array types
        """
        if isinstance(input, MutableJaxArray) or isinstance(input, MPIShared):
            # those types need to be cast into their inner data
            return input.data
        elif isinstance(input, np.ndarray) or isinstance(input, jax.numpy.ndarray):
            # those types can be feed to JAX raw with no error or performance problems
            return input
        else:
            # errors-out on other datatypes
            # so that we can make sure we are using the most efficient convertion available
            raise RuntimeError(f"Passed a {type(input)} to MutableJaxArray.to_array. Please find the best way to convert it to a Numpy array and update the function.")

    def to_numpy(self):
        """
        converts the content back to a numpy array
        we purposefully do not overload __array__ to avoid accidental conversions

        WARNING: this function will likely cost you a copy.
        """
        return jax.device_get(self.data)
    
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
        produces a new array with a different shape
        WARNING: this will copy the data and *not* propagate modifications to the older array
        """
        reshaped_data = jnp.reshape(self.data, newshape=shape)
        return MutableJaxArray(reshaped_data)
