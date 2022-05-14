# TODO
#
# In progress:
# - template_offset: ../templates/offset
#   the c++ version fails (tests 'template_offset' and 'ops_mapmaker_solve')
#   waiting for it to be fixed before merging the branch then trying to solve the problem
# Done:
# - pointing detector: pointing detector
# - build_noise_weighted: mapmaker_utils
# - stockes weight: stockes weigts
# - healpix_pixels: pixels_healpix
#
# - try merging intervals into a single indexes vector / mask (utils contains the needed functions) 
#   (we could cache that intervals_index at a higher level within toast if the intervals are constant)
# - try grouping intervals by size and then running with a (intervals_starts,length) input 
#   or take an upper bound on the intervals (starts, end-start+lengthmax), process the array thus constructed then later throw the parts that are not of interest
#   (only if the previous approach does not deliver)
# - try putting some template_offset operations in a jitted section 
#   (would be done by default if we can turn intervals into an index)
# - in pixels_healpix, does pixels matter or is hit_submaps the only real output?
# - update scan_map to slightly simplify it (cf associated TODO)
# - try to introduce donate_argnums in calls to xmap?
#
# - get rid of the self.use_python versions
# - report xmap plus static arg bug
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
