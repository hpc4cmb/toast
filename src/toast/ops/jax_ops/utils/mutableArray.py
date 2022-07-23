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
        # makes sure we are dealing with a device_put compatible input
        if isinstance(data, MPIShared):
            data = jnp.array(data)

        self.data = jax.device_put(data)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        self.nbytes = self.data.nbytes

    @classmethod
    def zeros(cls, shape, dtype=None):
        """creates an array of zeros"""
        data = jnp.zeros(shape=shape, dtype=dtype)
        return cls(data)

    def to_numpy(self):
        """
        converts the content back to a numpy array
        we purposefully do not overload __array__ to avoid accidental conversions

        WARNING: this function will likely cost you a copy.
        """
        return jax.device_get(self.data)
    
    def __setitem__(self, key, value):
        """replace the inner array in place"""
        self.data = self.data.at[key].set(value)

    def __getitem__(self, key):
        """access the inner array"""
        return self.data[key]
    
    def reshape(self, shape):
        """
        produces a new array with a different shape
        WARNING: this will copy the data and *not* propagate modifications to the older array
        """
        reshaped_data = jnp.reshape(self.data, newshape=shape)
        return MutableJaxArray(reshaped_data)