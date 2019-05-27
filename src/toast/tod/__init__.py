# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# import functions into our public API

from .tod import TOD, TODCache


from .interval import Interval

from .pointing import OpPointingHpix

from .sim_tod import (
    satellite_scanning,
    TODHpixSpiral,
    TODSatellite,
    slew_precession_axis,
    TODGround,
)

from .sim_noise import AnalyticNoise

from .sim_interval import regular_intervals

from .sim_det_noise import OpSimNoise

#
# from .sim_det_map import OpSimGradient, OpSimScan
#
# from .sim_det_dipole import OpSimDipole
#
# from .sim_det_atm import OpSimAtmosphere
#
# from .sim_focalplane import (
#     hex_layout,
#     rhombus_layout,
#     hex_pol_angles_qu,
#     hex_pol_angles_radial,
#     rhomb_pol_angles_qu,
#     plot_focalplane,
# )

from .noise import Noise

# from ..ctoast import filter_polyfilter
# from .polyfilter import OpPolyFilter
# from .groundfilter import OpGroundFilter
# from .gainscrambler import OpGainScrambler
from .applygain import OpApplyGain, write_calibration_file

# from .memorycounter import OpMemoryCounter

from .pointing_math import aberrate

from .tod_math import (
    calibrate,
    dipole,
    sim_noise_timestream,
    OpCacheCopy,
    OpCacheClear,
    flagged_running_average,
)

# from .conviqt import OpSimConviqt
#
# from .tidas import available as tidas_available
#
# from .spt3g_utils import available as spt3g_available
