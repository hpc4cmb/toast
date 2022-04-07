# TODO
# test C++, numpy and jax version of each operator:
# - pointing detector: pointing detector
# - stockes weight: stockes weigts
# - healpix_pixels: pixels_healpix
# - build_noise_weighted: mapmaker_utils
# - template_offset: ../templates/offset
#
# things to test for the interval loops:
# - try merging intervals into a single indexes vector
# - try grouping intervals by size and then running with a (intervals_starts,length) input
# - try putting some template_offset operations in a jitted section
#
# NOTE:
# the code uses [xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html) 
# in order to map over named axis for increased redeability
# however, one could use several vmap to reproduce this functionality

# enable 64bits precision
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)

# operators that have been ported to JAX
from .polyfilter1D import filter_polynomial
from .polyfilter2D import filter_poly2D
from .scan_map import scan_map
from .pixels_healpix import pixels_healpix
from .stokes_weights import stokes_weights_I, stokes_weights_IQU
from .template_offset import template_offset_add_to_signal, template_offset_project_signal, template_offset_apply_diag_precond
from .pointing_detector import pointing_detector
from .build_noise_weighted import build_noise_weighted
