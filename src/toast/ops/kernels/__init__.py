from ...accelerator import use_accel_jax

# FIXME: In a future cleanup, we should:
#
# 1. Name the kernel files after the operator files that use them.  This cleanup
#    was planned for the compiled kernels, but since the python / jax kernels have
#    adopted the same names, we need to clean those up too.
#
# 2. We can probably automate the namespace imports using attributes of the 
#    submodules.
#

from .compiled import build_noise_weighted as build_noise_weighted_compiled
from .compiled import cov_accum_diag_hits as cov_accum_diag_hits_compiled
from .compiled import cov_accum_diag_invnpp as cov_accum_diag_invnpp_compiled
from .compiled import filter_poly2D as filter_poly2D_compiled
from .compiled import filter_polynomial as filter_polynomial_compiled
from .compiled import noise_weight as noise_weight_compiled
from .compiled import pixels_healpix as pixels_healpix_compiled
from .compiled import pointing_detector as pointing_detector_compiled
from .compiled import scan_map as scan_map_compiled
from .compiled import stokes_weights_I as stokes_weights_I_compiled
from .compiled import stokes_weights_IQU as stokes_weights_IQU_compiled
from .compiled import (
    template_offset_add_to_signal as template_offset_add_to_signal_compiled,
)
from .compiled import (
    template_offset_apply_diag_precond as template_offset_apply_diag_precond_compiled,
)
from .compiled import (
    template_offset_project_signal as template_offset_project_signal_compiled,
)
from .implementation_selection import (
    ImplementationType,
    select_implementation,
    select_implementation_cpu,
)

if use_accel_jax:
    from .jax import build_noise_weighted as build_noise_weighted_jax
    from .jax import cov_accum_diag_hits as cov_accum_diag_hits_jax
    from .jax import cov_accum_diag_invnpp as cov_accum_diag_invnpp_jax
    from .jax import filter_poly2D as filter_poly2D_jax
    from .jax import filter_polynomial as filter_polynomial_jax
    from .jax import noise_weight as noise_weight_jax
    from .jax import pixels_healpix as pixels_healpix_jax
    from .jax import pointing_detector as pointing_detector_jax
    from .jax import scan_map as scan_map_jax
    from .jax import stokes_weights_I as stokes_weights_I_jax
    from .jax import stokes_weights_IQU as stokes_weights_IQU_jax
    from .jax import (
        template_offset_add_to_signal as template_offset_add_to_signal_jax,
    )
    from .jax import (
        template_offset_apply_diag_precond as template_offset_apply_diag_precond_jax,
    )
    from .jax import (
        template_offset_project_signal as template_offset_project_signal_jax,
    )
else:
    build_noise_weighted_jax = None
    cov_accum_diag_hits_jax = None
    cov_accum_diag_invnpp_jax = None
    filter_poly2D_jax = None
    filter_polynomial_jax = None
    noise_weight_jax = None
    pixels_healpix_jax = None
    pointing_detector_jax = None
    scan_map_jax = None
    stokes_weights_I_jax = None
    stokes_weights_IQU_jax = None
    template_offset_add_to_signal_jax = None
    template_offset_apply_diag_precond_jax = None
    template_offset_project_signal_jax = None

from .python import build_noise_weighted as build_noise_weighted_python
from .python import cov_accum_diag_hits as cov_accum_diag_hits_python
from .python import cov_accum_diag_invnpp as cov_accum_diag_invnpp_python
from .python import filter_poly2D as filter_poly2D_python
from .python import filter_polynomial as filter_polynomial_python
from .python import noise_weight as noise_weight_python
from .python import pixels_healpix as pixels_healpix_python
from .python import pointing_detector as pointing_detector_python
from .python import scan_map as scan_map_python
from .python import stokes_weights_I as stokes_weights_I_python
from .python import stokes_weights_IQU as stokes_weights_IQU_python
from .python import (
    template_offset_add_to_signal as template_offset_add_to_signal_python,
)
from .python import (
    template_offset_apply_diag_precond as template_offset_apply_diag_precond_python,
)
from .python import (
    template_offset_project_signal as template_offset_project_signal_python,
)

# kernels with use_accel
scan_map = select_implementation(scan_map_compiled, scan_map_python, scan_map_jax)
pixels_healpix = select_implementation(
    pixels_healpix_compiled, pixels_healpix_python, pixels_healpix_jax
)
stokes_weights_I = select_implementation(
    stokes_weights_I_compiled, stokes_weights_I_python, stokes_weights_I_jax
)
stokes_weights_IQU = select_implementation(
    stokes_weights_IQU_compiled, stokes_weights_IQU_python, stokes_weights_IQU_jax
)
template_offset_add_to_signal = select_implementation(
    template_offset_add_to_signal_compiled,
    template_offset_add_to_signal_python,
    template_offset_add_to_signal_jax,
)
template_offset_project_signal = select_implementation(
    template_offset_project_signal_compiled,
    template_offset_project_signal_python,
    template_offset_project_signal_jax,
)
template_offset_apply_diag_precond = select_implementation(
    template_offset_apply_diag_precond_compiled,
    template_offset_apply_diag_precond_python,
    template_offset_apply_diag_precond_jax,
)
pointing_detector = select_implementation(
    pointing_detector_compiled, pointing_detector_python, pointing_detector_jax
)
build_noise_weighted = select_implementation(
    build_noise_weighted_compiled, build_noise_weighted_python, build_noise_weighted_jax
)
noise_weight = select_implementation(
    noise_weight_compiled, noise_weight_python, noise_weight_jax
)

# kernels with no use_accel
cov_accum_diag_hits = select_implementation_cpu(
    cov_accum_diag_hits_compiled, cov_accum_diag_hits_python, cov_accum_diag_hits_jax
)
cov_accum_diag_invnpp = select_implementation_cpu(
    cov_accum_diag_invnpp_compiled,
    cov_accum_diag_invnpp_python,
    cov_accum_diag_invnpp_jax,
)
filter_polynomial = select_implementation_cpu(
    filter_polynomial_compiled, filter_polynomial_python, filter_polynomial_jax
)
filter_poly2D = select_implementation_cpu(
    filter_poly2D_compiled, filter_poly2D_python, filter_poly2D_jax
)

# kernels that have not been ported
from ..._libtoast import (
    accumulate_observation_matrix,
    add_templates,
    bin_invcov,
    bin_proj,
    build_template_covariance,
    cov_apply_diag,
    cov_eigendecompose_diag,
    expand_matrix,
    fod_autosums,
    fod_crosssums,
    fourier,
    healpix_pixels,
    integrate_simpson,
    legendre,
    stokes_weights,
    subtract_mean,
    sum_detectors,
    tod_sim_noise_timestream,
    tod_sim_noise_timestream_batch,
)
