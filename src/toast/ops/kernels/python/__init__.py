# operators that have been ported to Numpy

# operators ported but not using the use_accel input:
from .polyfilter1D import filter_polynomial
from .polyfilter2D import filter_poly2D
from .template_offset import (
    template_offset_add_to_signal,
    template_offset_apply_diag_precond,
    template_offset_project_signal,
)
