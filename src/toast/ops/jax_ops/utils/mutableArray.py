import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from pshmem import MPIShared
from ....utils import AlignedI64, AlignedF64

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
        data = MutableJaxArray.to_array(data)
        self.data = jax.device_put(data)

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
        Casts all array types used in TOAST to JAX-compatible array types
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
    
    def __str__(self):
        """
        returns a string representation of the content of the array
        """
        return self.data.__str__()

    def __eq__(self, other):
        raise RuntimeError("MutableJaxArray: tried an equality test on a MutableJaxArray. This container is not designed for computations, you likely have a data movement bug somewhere in your program.")