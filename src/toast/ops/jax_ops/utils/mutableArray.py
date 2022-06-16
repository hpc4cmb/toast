import jax.numpy as jnp
import numpy as np
from typing import Tuple

class MutableJaxArray():
    """
    This class encapsulate a jax array to give the illusion of mutability
    simplifying integration within toast
    """
    data: jnp.DeviceArray
    shape: Tuple
    dtype: np.dtype

    def __init__(self, data):
        """encapsulate an array as a jax array"""
        self.data = jnp.asarray(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    @classmethod
    def zeros(cls, shape):
        """creates an array of zeros"""
        data = jnp.zeros(shape=shape)
        return cls(data)

    def to_numpy(self):
        """
        converts the content back to a numpy array
        we purposefully do not overload __array__ to avoid accidental conversions
        """
        output = np.asarray(self.data)
        if not output.flags.writeable:
            # TODO needed because some array end up read-only
            print("DEBUGGING: read only")
            output = np.copy(output)
        else:
            print("DEBUGGING: readable") # TODO
        return output
    
    def __setitem__(self, key, value):
        """replace the inner array in place"""
        self.data = self.data.at[key].set(value)

    def __getitem__(self, key):
        """access the inner array"""
        return self.data[key]
    
    def reshape(self, shape):
        """changes the shape of the inner array"""
        reshaped_data = jnp.reshape(self.data, newshape=shape)
        return MutableJaxArray(reshaped_data)