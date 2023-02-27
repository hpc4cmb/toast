# NOTE:
# - the code uses [xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html)
#   in order to map over named axis for increased readability
#   however, one could use several vmap to reproduce this functionality

# enable 64bits precision
from jax.config import config as jax_config

jax_config.update("jax_enable_x64", True)
