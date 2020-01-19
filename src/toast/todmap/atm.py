# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import use_mpi

available_utils = None
if available_utils is None:
    available_utils = True
    try:
        from .._libtoast import (
            atm_absorption_coefficient,
            atm_absorption_coefficient_vec,
            atm_atmospheric_loading,
            atm_atmospheric_loading_vec,
        )
    except ImportError:
        available_utils = False

available = None
if available is None:
    available = True
    try:
        from .._libtoast import AtmSim
    except ImportError:
        available = False

available_mpi = None
if available_mpi is None:
    if use_mpi:
        available_mpi = True
        try:
            from .._libtoast_mpi import AtmSimMPI
        except ImportError:
            available_mpi = False
    else:
        available_mpi = False
