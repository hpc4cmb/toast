# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

from .accel import (
    use_accel_omp,
    use_accel_jax,
    accel_enabled,
    accel_assign_device,
    accel_get_device,
    accel_data_create,
    accel_data_present,
    accel_data_update_device,
    accel_data_update_host,
    accel_data_delete,
    AcceleratorObject,
)

from .kernel_registry import kernel, ImplementationType
