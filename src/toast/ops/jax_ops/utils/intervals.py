import numpy as np
import jax.numpy as jnp
from typing import NamedTuple

def build_jax_interval_dtype():
    """
    builds an interval type to be used inside jax arrays
    adapted from toast/intervals.py
    """
    dtdbl = jnp.dtype("double")
    dtll = jnp.dtype("longlong")
    fmts = [dtdbl.char, dtdbl.char, dtll.char, dtll.char]
    offs = [
        0,
        dtdbl.alignment,
        2 * dtdbl.alignment,
        2 * dtdbl.alignment + dtll.alignment,
    ]
    return jnp.dtype(
        {
            "names": ["start", "stop", "first", "last"],
            "formats": fmts,
            "offsets": offs,
        }
    )

jax_interval_dtype = build_jax_interval_dtype()
"""
interval type with a start, stop, first and last element
needed to export interval_dtype arrays to JAX
"""

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