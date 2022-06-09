import jax.numpy as jnp

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