# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""OpenCL tools.
"""

from .utils import (
    have_opencl,
    find_source,
    get_kernel_deps,
    add_kernel_deps,
    replace_kernel_deps,
    clear_kernel_deps,
)
from .platform import OpenCL
