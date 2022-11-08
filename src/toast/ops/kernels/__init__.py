from ...accelerator.implementation_selection import select_implementation

from .compiled_kernels import (
    scan_map as scan_map_compiled,
    pixels_healpix as pixels_healpix_compiled,
    stokes_weights_I as stokes_weights_I_compiled,
    stokes_weights_IQU as stokes_weights_IQU_compiled,
    template_offset_add_to_signal as template_offset_add_to_signal_compiled,
    template_offset_project_signal as template_offset_project_signal_compiled,
    template_offset_apply_diag_precond as template_offset_apply_diag_precond_compiled,
    pointing_detector as pointing_detector_compiled,
    build_noise_weighted as build_noise_weighted_compiled,
    noise_weight as noise_weight_compiled,
    cov_accum_diag_hits as cov_accum_diag_hits_compiled,
    cov_accum_diag_invnpp as cov_accum_diag_invnpp_compiled,
    filter_polynomial as filter_polynomial_compiled,
    filter_poly2D as filter_poly2D_compiled,
)

from .python_kernels import (
    scan_map as scan_map_python,
    pixels_healpix as pixels_healpix_python,
    stokes_weights_I as stokes_weights_I_python,
    stokes_weights_IQU as stokes_weights_IQU_python,
    template_offset_add_to_signal as template_offset_add_to_signal_python,
    template_offset_project_signal as template_offset_project_signal_python,
    template_offset_apply_diag_precond as template_offset_apply_diag_precond_python,
    pointing_detector as pointing_detector_python,
    build_noise_weighted as build_noise_weighted_python,
    noise_weight as noise_weight_python,
    cov_accum_diag_hits as cov_accum_diag_hits_python,
    cov_accum_diag_invnpp as cov_accum_diag_invnpp_python,
    filter_polynomial as filter_polynomial_python,
    filter_poly2D as filter_poly2D_python,
)

from .jax_kernels import (
    scan_map as scan_map_jax,
    pixels_healpix as pixels_healpix_jax,
    stokes_weights_I as stokes_weights_I_jax,
    stokes_weights_IQU as stokes_weights_IQU_jax,
    template_offset_add_to_signal as template_offset_add_to_signal_jax,
    template_offset_project_signal as template_offset_project_signal_jax,
    template_offset_apply_diag_precond as template_offset_apply_diag_precond_jax,
    pointing_detector as pointing_detector_jax,
    build_noise_weighted as build_noise_weighted_jax,
    noise_weight as noise_weight_jax,
    cov_accum_diag_hits as cov_accum_diag_hits_jax,
    cov_accum_diag_invnpp as cov_accum_diag_invnpp_jax,
    filter_polynomial as filter_polynomial_jax,
    filter_poly2D as filter_poly2D_jax,
)

# kernels with use_accel
scan_map = select_implementation(scan_map_compiled, scan_map_python, scan_map_jax)
pixels_healpix = select_implementation(pixels_healpix_compiled, pixels_healpix_python, pixels_healpix_jax)
stokes_weights_I = select_implementation(stokes_weights_I_compiled, stokes_weights_I_python, stokes_weights_I_jax)
stokes_weights_IQU = select_implementation(stokes_weights_IQU_compiled, stokes_weights_IQU_python, stokes_weights_IQU_jax)
template_offset_add_to_signal = select_implementation(template_offset_add_to_signal_compiled, template_offset_add_to_signal_python, template_offset_add_to_signal_jax)
template_offset_project_signal = select_implementation(template_offset_project_signal_compiled, template_offset_project_signal_python, template_offset_project_signal_jax)
template_offset_apply_diag_precond = select_implementation(template_offset_apply_diag_precond_compiled, template_offset_apply_diag_precond_python, template_offset_apply_diag_precond_jax)
pointing_detector = select_implementation(pointing_detector_compiled, pointing_detector_python, pointing_detector_jax)
build_noise_weighted = select_implementation(build_noise_weighted_compiled, build_noise_weighted_python, build_noise_weighted_jax)
noise_weight = select_implementation(noise_weight_compiled, noise_weight_python, noise_weight_jax)

# kernels with no use_accel, default to compiled
cov_accum_diag_hits = cov_accum_diag_hits_compiled
cov_accum_diag_invnpp = cov_accum_diag_invnpp_compiled
filter_polynomial = filter_polynomial_compiled
filter_poly2D = filter_poly2D_compiled

# kernels that have not been ported
from ..._libtoast import (
    add_templates, 
    bin_invcov, 
    bin_proj, 
    legendre,
    fourier,
    accumulate_observation_matrix,
    build_template_covariance,
    expand_matrix,
    fod_autosums, 
    fod_crosssums,
    subtract_mean, 
    sum_detectors,
    tod_sim_noise_timestream, 
    tod_sim_noise_timestream_batch,
    integrate_simpson,
    cov_apply_diag, 
    cov_eigendecompose_diag,
)
