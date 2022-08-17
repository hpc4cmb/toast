import jax
import numpy as np
import jax.numpy as jnp
from typing import NamedTuple

#------------------------------------------------------------------------------
# Numpy to JAX intervals

class INTERVALS_JAX(NamedTuple):
    """
    Encapsulate the information from an array of intervals in a JAX compatible way
    Using a struct of arrays rather than an array of structs.
    This class can be converted into a pytree (as it inherits from NamedTuple)
    """
    starts: np.float64
    stops: np.float64
    firsts: np.longlong
    lasts: np.longlong

    @classmethod
    def init(cls, data):
        starts = jax.device_put(data.start)
        stops = jax.device_put(data.stop)
        firsts = jax.device_put(data.first)
        lasts = jax.device_put(data.last)
        return cls(starts, stops, firsts, lasts)
    
    @classmethod
    def empty(cls):
        starts = jnp.empty(shape=0)
        stops = jnp.empty(shape=0)
        firsts = jnp.empty(shape=0)
        lasts = jnp.empty(shape=0)
        return cls(starts, stops, firsts, lasts)
    
    def to_numpy(self, output_buffer):
        """copies data back into the given buffer"""
        output_buffer.start[:] = self.starts
        output_buffer.stop[:] = self.stops
        output_buffer.first[:] = self.firsts
        output_buffer.last[:] = self.lasts

#------------------------------------------------------------------------------
# Irregular intervals

ALL = slice(None,None,None)
"""
Full slice, equivalent to `:` in `[:]`.
"""

class JaxIntervals:
    """
    Class to helps dealing with variable-size intervals in JAX.
    Internally, it pads the data to max_length on read and masks them on write.
    WARNING: this function is designed to be used inside a jitted-function as it can be very memory hungry otherwise.
    """
    def __init__(self, starts, ends, max_length):
        """
        Builds a JaxIntervals object using the starting and ending points of all intervals
        plus the length of the larger interval (this needs to be a static quantity).
        """
        # 2D tensor of integer of shape (nb_interval,max_length)
        self.indices = starts[:,jnp.newaxis] + jnp.arange(max_length)
        # mask that is True on all values that should be ignored in the interval
        self.mask = self.indices >= ends[:,jnp.newaxis]

    def _interval_of_key(key):
        """
        Takes a key, expected to be a JaxIntervals or a tuple with at least one JaxIntervals member
        and returns (key, mask) where both key and mask can be used on a matrix that would be indexed by the original key.
        """
        # insures that the key is a tuple
        if not isinstance(key, tuple): key = (key,)
        # insures all elements of the key are valid
        # and finds the interval
        mask = None
        def fix_key(key):
            nonlocal mask
            if isinstance(key, JaxIntervals):
                # stores the interval and returns the actual index
                mask = key.mask
                return key.indices
            elif not (mask is None):
                # adds a trailing dimension to the mask (as there are subsequent dimenssions)
                mask = jnp.expand_dims(mask, axis=-1)
            # adds two trailing dimensions to arrays, one per interval dimenssion
            if isinstance(key, jnp.ndarray) or isinstance(key, np.ndarray):
                return key[:, jnp.newaxis, jnp.newaxis]
            return key
        key = tuple(fix_key(k) for k in key)
        # makes sure at least one of the indices was an interval
        if mask is None:
            raise RuntimeError("JaxIntervals: your key should contain a JaxIntervals type.")
        return (key, mask)

    def get(data, key, padding_value=None):
        """
        Gets the data at the given key
        the result will be padded to keep the interval size constant
        we will use values from data to pad unless padding_value is not None
        we expect key to be a JaxIntervals or a tuple with at least one JaxIntervals member.
        """
        key, mask = JaxIntervals._interval_of_key(key)
        return data[key] if (padding_value is None) else jnp.where(mask, padding_value, data[key])

    def set(data, key, value_intervals):
        """
        Sets the data at the given key with value_intervals
        we expect key to be a JaxIntervals or a tuple with at least one JaxIntervals member.
        """
        key, mask = JaxIntervals._interval_of_key(key)
        value_interval_masked = jnp.where(mask, data[key], value_intervals)
        return data.at[key].set(value_interval_masked)
