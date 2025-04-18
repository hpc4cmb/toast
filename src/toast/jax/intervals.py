import jax
import numpy as np

from ..timing import function_timer


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
            return np.max(intervals.last - intervals.first)

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
