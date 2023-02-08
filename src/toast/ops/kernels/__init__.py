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
    integrate_simpson,
    legendre,
    subtract_mean,
    sum_detectors,
    tod_sim_noise_timestream,
    tod_sim_noise_timestream_batch,
)
from ...accelerator import use_accel_jax
from .implementation_selection import (
    ImplementationType,
    select_implementation,
    select_implementation_cpu,
)

# FIXME: In a future cleanup, we should:
#
# 1. Name the kernel files after the operator files that use them.  This cleanup
#    was planned for the compiled kernels, but since the python / jax kernels have
#    adopted the same names, we need to clean those up too.
#
# 2. We can probably automate the namespace imports using attributes of the
#    submodules.
#


# if use_accel_jax:
#     from .jax import (
#         template_offset_add_to_signal as template_offset_add_to_signal_jax,
#     )
#     from .jax import (
#         template_offset_apply_diag_precond as template_offset_apply_diag_precond_jax,
#     )
#     from .jax import (
#         template_offset_project_signal as template_offset_project_signal_jax,
#     )
# else:
#     template_offset_add_to_signal_jax = None
#     template_offset_apply_diag_precond_jax = None
#     template_offset_project_signal_jax = None

# from .python import (
#     template_offset_add_to_signal as template_offset_add_to_signal_python,
# )
# from .python import (
#     template_offset_apply_diag_precond as template_offset_apply_diag_precond_python,
# )
# from .python import (
#     template_offset_project_signal as template_offset_project_signal_python,
# )

# kernels with use_accel

# template_offset_add_to_signal = select_implementation(
#     template_offset_add_to_signal_compiled,
#     template_offset_add_to_signal_python,
#     template_offset_add_to_signal_jax,
# )
# template_offset_project_signal = select_implementation(
#     template_offset_project_signal_compiled,
#     template_offset_project_signal_python,
#     template_offset_project_signal_jax,
# )
# template_offset_apply_diag_precond = select_implementation(
#     template_offset_apply_diag_precond_compiled,
#     template_offset_apply_diag_precond_python,
#     template_offset_apply_diag_precond_jax,
# )

# kernels with no use_accel
