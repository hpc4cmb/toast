# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from datetime import datetime

import numpy as np

from .mpi import MPITestCase

import numpy.testing as nt

from ..weather import Weather, SimWeather


class WeatherTest(MPITestCase):
    def setUp(self):
        pass

    def test_sim(self):
        date = datetime.now()

        sim_atacama = SimWeather(time=date, name="atacama", site_uid=1)
        sim_pole = SimWeather(time=date, name="south_pole", site_uid=2)

        # for name, sim in zip(["atacama", "south_pole"], [sim_atacama, sim_pole]):
        #     print(
        #         "{} (uid = {}, realiz. = {}):".format(
        #             name, sim._site_uid, sim._realization
        #         )
        #     )
        #     print("  time = ", sim.time)
        #     print("  ice_water = ", sim.ice_water)
        #     print("  liquid_water = ", sim.liquid_water)
        #     print("  pwv = ", sim.pwv)
        #     print("  humidity = ", sim.humidity)
        #     print("  surface_pressure = ", sim.surface_pressure)
        #     print("  surface_temperature = ", sim.surface_temperature)
        #     print("  air_temperature = ", sim.air_temperature)
        #     print("  west_wind = ", sim.west_wind)
        #     print("  south_wind = ", sim.south_wind)
