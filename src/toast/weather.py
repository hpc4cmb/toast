# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

from collections import OrderedDict

import datetime

from pkg_resources import resource_filename

import numpy as np

from astropy import units as u

import h5py

from . import rng as rng

from .timing import function_timer


class Weather(object):
    """Base class representing the weather for one observation.

    This class returns the nominal weather properties for an observation.  Currently
    these are constant, under the assumption that the weather changes slowly during a
    good science observation.

    """

    def __init__(self):
        pass

    def _time(self):
        raise NotImplementedError("Derived class must implement _time()")

    @property
    def time(self):
        """The time for these weather properties."""
        return self._time()

    def _ice_water(self):
        raise NotImplementedError("Derived class must implement _ice_water()")

    @property
    def ice_water(self):
        """Total precipitable ice water [kg/m^2] (also [mm])."""
        return self._ice_water()

    def _liquid_water(self):
        raise NotImplementedError("Derived class must implement _liquid_water()")

    @property
    def liquid_water(self):
        """Total precipitable liquid water [kg/m^2] (also [mm])."""
        return self._liquid_water()

    def _pwv(self):
        raise NotImplementedError("Derived class must implement _pwv()")

    @property
    def pwv(self):
        """Total precipitable water vapor [kg/m^2] (also [mm])."""
        return self._pwv()

    def _humidity(self):
        raise NotImplementedError("Derived class must implement _humidity()")

    @property
    def humidity(self):
        """10-meter specific humidity [kg/kg]"""
        return self._humidity()

    def _surface_pressure(self):
        raise NotImplementedError("Derived class must implement _surface_pressure()")

    @property
    def surface_pressure(self):
        """Surface pressure [Pa]."""
        return self._surface_pressure()

    def _surface_temperature(self):
        raise NotImplementedError("Derived class must implement _surface_temperature()")

    @property
    def surface_temperature(self):
        """Surface skin temperature [K]."""
        return self._surface_temperature()

    def _air_temperature(self):
        raise NotImplementedError("Derived class must implement _air_temperature()")

    @property
    def air_temperature(self):
        """10-meter air temperature [K]."""
        return self._air_temperature()

    def _west_wind(self):
        raise NotImplementedError("Derived class must implement _west_wind()")

    @property
    def west_wind(self):
        """10-meter eastward wind [m/s]."""
        return self._west_wind()

    def _south_wind(self):
        raise NotImplementedError("Derived class must implement _south_wind()")

    @property
    def south_wind(self):
        """10-meter northward wind [m/s]."""
        return self._south_wind()


def read_weather(file):
    """Helper function to read HDF5 format weather file.

    Args:
        file (str):  Path to the file.

    Returns:
        (dict):  Weather data dictionary.

    """
    hf = h5py.File(file, "r")
    result = OrderedDict()
    for mn in range(12):
        month_data = OrderedDict()
        month = "month_{:02d}".format(mn)
        meta = hf[month].attrs
        for k, v in meta.items():
            month_data[k] = v
        # Build the index of the distribution
        month_data["prob"] = np.linspace(
            month_data["PROBSTRT"],
            month_data["PROBSTOP"],
            month_data["NSTEP"],
        )
        # Iterate over datasets, copying to regular numpy arrays
        month_data["data"] = OrderedDict()
        for dname, dat in hf[month].items():
            month_data["data"][dname] = np.array(dat)
        result[mn] = month_data
    hf.close()
    return result


package_weather_data = None


def load_package_weather(name):
    """Helper function to read and cache bundled weather files.

    Args:
        name (str):  The file name (without the .h5 suffix)

    Returns:
        (dict):  The weather data for the specified site.

    """
    global package_weather_data

    if package_weather_data is None:
        package_weather_data = dict()

    if name in package_weather_data:
        return package_weather_data[name]

    weather_dir = resource_filename("toast", os.path.join("aux", "weather"))
    weather_file = os.path.join(weather_dir, "{}.h5".format(name))
    if not os.path.isfile(weather_file):
        msg = "package weather file {} does not exist".format(weather_file)
        raise RuntimeError(msg)

    package_weather_data[name] = read_weather(weather_file)

    return package_weather_data[name]


class SimWeather(Weather):
    """Simulated weather properties based on historical data.

    The weather parameter distributions are read from site-specific weather files.
    The files contain parameter distributions for every UTC hour of the day, averaged
    over months.

    Supported name values (using a bundled file) are currently:
        - "atacama"
        - "south_pole"

    Alternatively a file path may be provided.

    Args:
        time (datetime):  A python date/time in UTC.
        name (str):  Supported name of the site.
        file (str):  Alternative file to load in the same format as the bundled data.
        site_uid (int):  The Unique ID for the site, used for random draw of parameters.
        realization (int):  The realization index used for random draw of parameters.

    """

    def __init__(self, time=None, name=None, file=None, site_uid=0, realization=0):
        if time is None:
            raise RuntimeError("you must specify the time")
        self._name = name
        if self._name is None:
            if file is None:
                raise RuntimeError("you must specify a name or file")
            else:
                self._data = read_weather(file)
                self._name = file
        else:
            self._data = load_package_weather(self._name)

        self._site_uid = site_uid
        self._realization = realization

        self._date = time
        self._doy = self._date.timetuple().tm_yday
        self._year = self._date.year
        self._hour = self._date.hour
        # This is the definition of month used in the weather files
        self._month = int((self._doy - 1) // 30.5)

        # Use a separate RNG index for each data type
        self._varindex = {y: x for x, y in enumerate(self._data[0]["data"].keys())}

        self._sim_ice_water = self._draw("TQI")
        self._sim_liquid_water = self._draw("TQL")
        self._sim_pwv = self._draw("TQV")
        self._sim_humidity = self._draw("QV10M")
        self._sim_surface_pressure = self._draw("PS")
        self._sim_surface_temperature = self._draw("TS") * u.Kelvin
        self._sim_air_temperature = self._draw("T10M") * u.Kelvin
        self._sim_west_wind = self._draw("U10M") * (u.meter / u.second)
        self._sim_south_wind = self._draw("V10M") * (u.meter / u.second)
        super().__init__()

    def _draw(self, name):
        """Return a random parameter value.

        Return a random value for preset variable and time.

        Args:
            name(str): MERRA-2 name for the variable.

        Returns:
            (float): The parameter value.

        """
        # Set the RNG counters for this variable and time
        counter1 = self._varindex[name]
        counter2 = (self._year * 366 + self._doy) * 24 + self._hour

        # Get a uniform random number for inverse sampling
        x = rng.random(
            1,
            sampler="uniform_01",
            key=(self._site_uid, self._realization),
            counter=(counter1, counter2),
        )[0]

        # Sample the variable from the inverse cumulative distribution function
        prob = self._data[self._month]["prob"]
        cdf = self._data[self._month]["data"][name][self._hour]
        return np.interp(x, prob, cdf)

    def _time(self):
        return self._date

    def _ice_water(self):
        return self._sim_ice_water

    def _liquid_water(self):
        return self._sim_liquid_water

    def _pwv(self):
        return self._sim_pwv

    def _humidity(self):
        return self._sim_humidity

    def _surface_pressure(self):
        return self._sim_surface_pressure

    def _surface_temperature(self):
        return self._sim_surface_temperature

    def _air_temperature(self):
        return self._sim_air_temperature

    def _west_wind(self):
        return self._sim_west_wind

    def _south_wind(self):
        return self._sim_south_wind

    def __repr__(self):
        value = "<SimWeather : '{}', year = {}, month = {}, hour = {}, site UID = {}, realization = {})".format(
            self._name,
            self._year,
            self._month,
            self._hour,
            self._site_uid,
            self._realization,
        )
        return value
