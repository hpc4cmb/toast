import jax
import numpy as np
import jax.numpy as jnp
from typing import NamedTuple

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