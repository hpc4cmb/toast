# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from datetime import datetime

import astropy.units as u
import numpy as np
import numpy.testing as nt

from ..weather import SimWeather, Weather
from .mpi import MPITestCase


class WeatherTest(MPITestCase):
    def setUp(self):
        pass

    def get_props(self, w):
        val = w.time
        val = w.ice_water
        val = w.liquid_water
        val = w.pwv
        val = w.humidity
        val = w.surface_pressure
        val = w.surface_temperature
        val = w.air_temperature
        val = w.west_wind
        val = w.south_wind

    def test_base(self):
        date = datetime.now()
        real = Weather(
            time=date,
            ice_water=1.0e-4 * u.mm,
            liquid_water=1.0e-4 * u.mm,
            pwv=2.0 * u.mm,
            humidity=0.005 * u.mm,
            surface_pressure=53000 * u.Pa,
            surface_temperature=273.0 * u.Kelvin,
            air_temperature=270.0 * u.Kelvin,
            west_wind=2.0 * (u.meter / u.second),
            south_wind=1.0 * (u.meter / u.second),
        )
        self.get_props(real)

    def test_sim(self):
        date = datetime.now()

        sim_atacama = SimWeather(time=date, name="atacama", site_uid=1)
        self.get_props(sim_atacama)

        sim_pole = SimWeather(time=date, name="south_pole", site_uid=2)
        self.get_props(sim_pole)
