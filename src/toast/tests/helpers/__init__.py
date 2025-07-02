# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# Namespace imports

from .flags import fake_flags
from .ground import (
    create_ground_data,
    create_ground_telescope,
    create_overdistributed_data,
)
from .hwp import fake_hwpss, fake_hwpss_data
from .sky import (
    create_fake_beam_alm,
    create_fake_healpix_file,
    create_fake_healpix_map,
    create_fake_healpix_scanned_tod,
    create_fake_mask,
    create_fake_sky_alm,
    create_fake_wcs_file,
    create_fake_wcs_map,
    create_fake_wcs_scanned_tod,
    fetch_nominal_cmb_cls,
)
from .space import (
    create_boresight_telescope,
    create_healpix_ring_satellite,
    create_satellite_data,
    create_satellite_data_big,
    create_satellite_empty,
    create_satellite_schedule,
    create_space_telescope,
)
from .utils import close_data, create_comm, create_outdir, uniform_chunks
