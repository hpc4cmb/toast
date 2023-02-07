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

from .compiled import filter_poly2D as filter_poly2D_compiled
from .compiled import filter_polynomial as filter_polynomial_compiled
from .compiled import noise_weight as noise_weight_compiled
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
    from .jax import filter_poly2D as filter_poly2D_jax
    from .jax import filter_polynomial as filter_polynomial_jax
    from .jax import noise_weight as noise_weight_jax
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
    filter_poly2D_jax = None
    filter_polynomial_jax = None
    noise_weight_jax = None
    template_offset_add_to_signal_jax = None
    template_offset_apply_diag_precond_jax = None
    template_offset_project_signal_jax = None

from .python import filter_poly2D as filter_poly2D_python
from .python import filter_polynomial as filter_polynomial_python
from .python import noise_weight as noise_weight_python
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
noise_weight = select_implementation(
    noise_weight_compiled, noise_weight_python, noise_weight_jax
)

# kernels with no use_accel
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
    integrate_simpson,
    legendre,
    subtract_mean,
    sum_detectors,
    tod_sim_noise_timestream,
    tod_sim_noise_timestream_batch,
)
