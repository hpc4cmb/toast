# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import types

import numpy as np
import traitlets
from astropy import units as u
from scipy.optimize import Bounds, curve_fit, least_squares

from ..mpi import MPI
from ..noise import Noise
from ..noise_sim import AnalyticNoise
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger
from ..weather import SimWeather
from .operator import Operator


@trait_docs
class WeatherModel(Operator):
    """Create a default weather model

    The weather model is used to draw observing conditions such as
    temperature, wind and PWV.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    weather = Unicode(
        None,
        allow_none=True,
        help="Name of built-in weather site (e.g. 'atacama', 'south_pole') or path to HDF5 file",
    )

    realization = Int(0, help="The realization index")

    max_pwv = Quantity(
        None, allow_none=True, help="Maximum PWV for the simulated weather."
    )

    median_weather = Bool(
        False,
        help="Use median weather parameters instead of sampling from the distributions",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in ("weather",):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        for ob in data.obs:
            comm = data.comm.comm_group
            site = ob.telescope.site
            times = ob.shared[self.times]
            tmin = times[0]
            tmax = times[-1]
            if comm is not None:
                tmin = comm.allreduce(tmin, MPI.MIN)
                tmax = comm.allreduce(tmax, MPI.MAX)
            from datetime import datetime, timezone

            mid_time = datetime.fromtimestamp((tmin + tmax) / 2, timezone.utc)
            try:
                weather = SimWeather(
                    time=mid_time,
                    name=self.weather,
                    site_uid=site.uid,
                    realization=self.realization,
                    max_pwv=self.max_pwv,
                    median_weather=self.median_weather,
                )
            except RuntimeError:
                # must be a file
                weather = SimWeather(
                    time=mid_time,
                    file=self.weather,
                    site_uid=site.uid,
                    realization=self.realization,
                    max_pwv=self.max_pwv,
                    median_weather=self.median_weather,
                )
            site.weather = weather

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {"shared": [self.times]}
        return

    def _provides(self):
        prov = {"meta": []}
        return prov
