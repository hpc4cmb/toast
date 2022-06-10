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

    def __init__(self, data):
        """encapsulate an array as a jax array"""
        self.data = jnp.asarray(data)
        self.shape = data.shape

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
        return np.asarray(self.data)
    
    def __setitem__(self, key, value):
        """replace the inner array in place"""
        self.data = self.data.at[key].set(value)

    def __getitem__(self, key):
        """access the inner array"""
        return self.data[key]