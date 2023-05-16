import jax
import jax.numpy as jnp
import numpy as np

from ..timing import function_timer

# TODO we need to clean up the two very similar interval type names

# ------------------------------------------------------------------------------
# Numpy to JAX intervals


class INTERVALS_JAX:
    """
    Encapsulate the information from an array of structures into a structure of array
    each of the inner arrays is converted into a JAX array
    """

    @function_timer
    def __init__(self, data):
        # otherwise
        self.size = data.size
        self.intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(data)
        self.start = jax.device_put(data.start)
        self.stop = jax.device_put(data.stop)
        self.first = jax.device_put(data.first)
        self.last = jax.device_put(data.last)
        # sets Numpy buffer aside in case we want to return it
        self.host_data = data.host_data if isinstance(data, INTERVALS_JAX) else data

    def compute_max_intervals_length(intervals):
        """
        given an array of intervals, returns the maximum interval length
        this function will use a precomputed values when using an INTERVALS_JAX
        """
        if isinstance(intervals, INTERVALS_JAX):
            return intervals.intervals_max_length
        elif intervals.size == 0:
            return 1
        else:
            # end+1 as TOAST intervals are inclusive
            return 1 + np.max(intervals.last - intervals.first)

    @function_timer
    def to_host(self):
        """copies data back into the original buffer and returns it"""
        self.host_data.start[:] = self.start
        self.host_data.stop[:] = self.stop
        self.host_data.first[:] = self.first
        self.host_data.last[:] = self.last
        return self.host_data

    def __iter__(self):
        # NOTE: this is only correct if intervals are write only
        return iter(self.host_data)

    def __len__(self):
        return self.size


# ------------------------------------------------------------------------------
# Irregular intervals

ALL = slice(None, None, None)
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
        self.indices = starts[:, jnp.newaxis] + jnp.arange(max_length)
        # mask that is True on all values that should be ignored in the interval
        self.mask = self.indices >= ends[:, jnp.newaxis]

    def _interval_of_key(key):
        """
        Takes a key, expected to be a JaxIntervals or a tuple with at least one JaxIntervals member
        and returns (key, mask) where both key and mask can be used on a matrix that would be indexed by the original key.
        """
        # insures that the key is a tuple
        if not isinstance(key, tuple):
            key = (key,)
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
            if (isinstance(key, jnp.ndarray) or isinstance(key, np.ndarray)) and (
                key.ndim > 0
            ):
                return key[:, jnp.newaxis, jnp.newaxis]
            return key

        key = tuple(fix_key(k) for k in key)
        # makes sure at least one of the indices was an interval
        if mask is None:
            raise RuntimeError(
                "JaxIntervals: your key should contain a JaxIntervals type."
            )
        return (key, mask)

    def get(data, key, padding_value=None):
        """
        Gets the data at the given key
        the result will be padded to keep the interval size constant
        we will use values from data to pad unless padding_value is not None
        we expect key to be a JaxIntervals or a tuple with at least one JaxIntervals member.

        NOTE:
        if you want to check whether padded data ends up in your final result, you can set `padding_value=np.nan` when testing
        (do avoid nan in production as it will significantly decrease the GPU performance)
        """
        key, mask = JaxIntervals._interval_of_key(key)
        return (
            data[key]
            if (padding_value is None)
            else jnp.where(mask, padding_value, data[key])
        )

    def set(data, key, value_intervals):
        """
        Sets the data at the given key with value_intervals
        we expect key to be a JaxIntervals or a tuple with at least one JaxIntervals member.
        """
        key, mask = JaxIntervals._interval_of_key(key)
        value_interval_masked = jnp.where(mask, data[key], value_intervals)
        return data.at[key].set(value_interval_masked)
