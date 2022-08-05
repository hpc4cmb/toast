import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from pshmem import MPIShared

#----------------------------------------------------------------------------------------
# In-place operations

def to_start_inner(key):
    """
    converts a slice or integer index into its start
    """
    if isinstance(key, slice):
        return 0 if (key.start is None) else key.start
    elif isinstance(key, np.ndarray):
        raise RuntimeError("to_start_inner: you cannot used this function with array indices, please use a workaround.")
    else:
        # NOTE: casts to int to avoid an int32 vs int64 discrepency with the slice
        return int(key)

def to_start(key, shape):
    """
    Converts an array index into its starting positions
    and ensures that the output is a tuple of the same size as the shape
    NOTE: this cannot be jitted
    """
    # ensures the key is a tuple
    if not isinstance(key, tuple): 
        key = (key,)

    # builds the output
    # if shape has more elements than key, we will default to 0 for those elements
    output = [0] * len(shape)
    for i,k in enumerate(key):
        output[i] = to_start_inner(k)
    
    # converts back to a tuple
    return tuple(output)

def to_size_inner(key,fullsize):
    """
    converts a slice or integer index into its size
    """
    if isinstance(key, slice):
        start = 0 if (key.start is None) else key.start
        stop = fullsize if (key.stop is None) else key.stop
        return stop - start
    else:
        return 1

def to_start_size(key,shape):
    """
    Converts an array index into its starts, sizes
    and ensures that the output is a tuple of the same size as the shape
    NOTE: this cannot be jitted
    """
    # ensures the key is a tuple
    if not isinstance(key, tuple): 
        key = (key,)

    # builds the output
    # if shape has more elements than key, we will default to:
    # - shape elements for size
    # - 0 elements for start
    output_start = [0] * len(shape)
    output_size = list(shape)
    for i,k in enumerate(key):
        fullsize = shape[i]
        output_start[i] = to_start_inner(k)
        output_size[i] = to_size_inner(k,fullsize)
    
    # converts back to a tuple
    output_start = tuple(output_start)
    output_size = tuple(output_size)
    return output_start, output_size

#-----

def update_in_place(data, value, start):
    """
    Performs data[key] = value where key is deduced from start and value.shape
    jit-compiled with buffer donation to ensure that the operation is done in place
    NOTE: as this function is jitted, you don't want to call it with different sizes every time
    WARNING: this function will fail if one of the arguments is an array
    """
    print(f"DEBUG: jit-compiling 'update_in_place' for data:{data.shape} value:{value.shape}")
    # insures that value has the same rank as data
    nb_missing_dims = data.ndim - value.ndim
    expanded_shape = (1,) * nb_missing_dims + value.shape
    value = jnp.reshape(value, expanded_shape)
    # insures that value has the same type as data
    value = value.astype(data.dtype)
    # does the inplace update
    return jax.lax.dynamic_update_slice(data, value, start)
update_in_place = jax.jit(fun=update_in_place, donate_argnums=[0])

def get_jitted(data, start, size):
    """
    Returns data[key] where key is deduced from start and value.shape
    jit-compiled for improved performance
    NOTE: as this function is jitted, you don't want to call it with different sizes every time
    WARNING: this function will fail if one of the arguments is an array
    """
    print(f"DEBUG: jit-compiling 'get_jitted' for data:{data.shape} size:{size}")
    return jax.lax.dynamic_slice(data, start, size)
get_jitted = jax.jit(fun=get_jitted, static_argnames=['size'])

def reorder_by_index_jitted(data, indices):
    """
    Computes data[indices,:,:] = data, returns data
    jit-compiled with buffer donation to ensure that the operation is done in place
    """
    print(f"DEBUG: jit-compiling 'reorder_by_index_jitted' for data:{data.shape}")
    if data.ndim == 1:
        return data.at[indices].set(data)
    elif data.ndim == 2:
        return data.at[indices,:].set(data)
    elif data.ndim == 3:
        return data.at[indices,:,:].set(data)
    else:
        raise RuntimeError("reorder_by_index_jitted: function needs updating for data with more than 3 dimensions.")
reorder_by_index_jitted = jax.jit(fun=reorder_by_index_jitted, donate_argnums=[0])

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
        """
        updates the inner array in place
        """
        #self.data = self.data.at[key].set(value)
        start = to_start(key, self.shape)
        self.data = update_in_place(self.data, value, start)

    def __getitem__(self, key):
        """
        access the inner array
        NOTE: surprisingly, this benefits from jitting
        """
        return self.data[key]
        # TODO the alternative is faster but causes an out-of-memory error somewhere...
        #start, size = to_start_size(key, self.shape)
        #return get_jitted(self.data, start, size)
    
    def reshape(self, shape):
        """
        produces a new array with a different shape
        WARNING: this will copy the data and *not* propagate modifications to the older array
        """
        reshaped_data = jnp.reshape(self.data, newshape=shape)
        return MutableJaxArray(reshaped_data)
