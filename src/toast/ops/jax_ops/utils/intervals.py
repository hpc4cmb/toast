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
        starts = data.start
        stops = data.stop
        firsts = data.first
        lasts = data.last
        return cls(starts, stops, firsts, lasts)
    
    @classmethod
    def empty(cls):
        starts = jnp.empty(shape=0)
        stops = jnp.empty(shape=0)
        firsts = jnp.empty(shape=0)
        lasts = jnp.empty(shape=0)
        return cls(starts, stops, firsts, lasts)