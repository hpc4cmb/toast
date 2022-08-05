import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from pshmem import MPIShared

#----------------------------------------------------------------------------------------
# In-place operations

def to_start_inner(key):
    """
    converts a slice into its begining
    returns everything else untouched
    """
    if isinstance(key, slice):
        return 0 if (key.start is None) else key.start
    else:
        return key

def to_start(key):
    """
    Converts an array index into its starting positions
    and ensures that the output is a tuple
    NOTE: this cannot be jitted
    """
    if isinstance(key, tuple):
        # ensures the key is a tuple
        return (to_start_inner(key),)
    else:
        # iterates on all elements of a tuple
        return tuple(to_start_inner(k) for k in key)

def update_in_place(data, value, start):
    """
    Performs data[key] = value
    jit-compiled with buffer donation to ensure that the operation is done in place
    NOTE: as this function is jitted, you don't want to call it with a wide variety of sizes
    """
    return jax.lax.dynamic_update_slice(data, value, start)
update_in_place = jax.jit(fun=update_in_place, donate_argnums=[0])

#----------------------------------------------------------------------------------------
# Mutable array

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
            raise RuntimeError(f"Passed a {type(data)} to MutableJaxArray. Please find the best way to convert it to a Numpy array.")

        # gets basic information on the data
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
        """updates the inner array in place"""
        #self.data = self.data.at[key].set(value)
        start = to_start(key)
        self.data = update_in_place(self.data, value, start)

    def __getitem__(self, key):
        """access the inner array"""
        return self.data[key]
    
    def reshape(self, shape):
        """
        produces a new array with a different shape
        WARNING: this will copy the data and *not* propagate modifications to the older array
        TODO: would it be sensible to do this operation in place?
        """
        reshaped_data = jnp.reshape(self.data, newshape=shape)
        return MutableJaxArray(reshaped_data)
