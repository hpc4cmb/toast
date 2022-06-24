# TODO
#
# to run miniapp:
# TOAST_GPU_JAX=true TOAST_LOGLEVEL=DEBUG toast_mini --node_mem_gb 4.0
# export TOAST_GPU_JAX=false; export JAX_PLATFORM_NAME=gpu; export OMP_NUM_THREADS=32; timer toast_mini --node_mem_gb 4.0
# TOAST_GPU_JAX=false JAX_PLATFORM_NAME=gpu nsys profile --stats=true toast_mini --node_mem_gb 4.0
#
# TODO: check that the pixel domain data is getting back to the host at the end of build_noise_weighted
#
# - pointing_detector runs in accel false but getting a jaxmutablearray!
#   add warning to kernels running on gpu mode but getting only cpu data (input | output) and vice versa (cpu mode but some gpu data)
#
# - fix circular import problem in accelerator.py (currently using a ugly fix)
#   ImportError: cannot import name 'import_from_name' from partially initialized module 'toast.utils' (most likely due to a circular import)
#
# - try putting some template_offset operations in a jitted section 
#   (would be done by default if we can turn intervals into an index)
#
# - update scan_map to slightly simplify it (cf associated TODO)
#
# - get rid of the self.use_python versions
#
# Alternative to the loop on interval:
# - try merging intervals into a single indexes vector / mask (utils contains the needed functions) 
#   (we could cache that intervals_index at a higher level within toast if the intervals are constant)
# - try grouping intervals by size and then running with a (intervals_starts,length) input 
#   or take an upper bound on the intervals (starts, end-start+lengthmax), process the array thus constructed then later throw the parts that are not of interest
#   (only if the previous approach does not deliver)
#
# NOTE:
# the code uses [xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html) 
# in order to map over named axis for increased readability
# however, one could use several vmap to reproduce this functionality

# enable 64bits precision
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)

# Jax types
from .utils import MutableJaxArray

# operators that have been ported to JAX
from .polyfilter1D import filter_polynomial
from .polyfilter2D import filter_poly2D
from .scan_map import scan_map
from .pixels_healpix import pixels_healpix
from .stokes_weights import stokes_weights_I, stokes_weights_IQU
from .template_offset import template_offset_add_to_signal, template_offset_project_signal, template_offset_apply_diag_precond
from .pointing_detector import pointing_detector
from .build_noise_weighted import build_noise_weighted
from .cov_accum import cov_accum_diag_hits, cov_accum_diag_invnpp
