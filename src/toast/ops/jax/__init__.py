from jax.config import config as jax_config
# enable 64bits precision
jax_config.update("jax_enable_x64", True)

from .utils import get_compile_time
from .polyfilter1D import filter_polynomial
from .polyfilter2D import filter_poly2D