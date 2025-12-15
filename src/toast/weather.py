# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import datetime
import os
from collections import OrderedDict
from datetime import timezone
from importlib import resources

import h5py
import numpy as np
from astropy import units as u

from . import rng as rng
from .timing import function_timer
from .utils import Logger, name_UID, hdf5_use_serial, import_from_name, object_fullname


class Weather(object):
    """Base class representing the weather for one observation.

    This class returns the nominal weather properties for an observation.  Currently
    these are constant, under the assumption that the weather changes slowly during a
    good science observation.

    This base class can be used directly to hold values specified at construction.

    Args:
        time (datetime):  A python date/time in UTC.
        ice_water (Quantity):  Precipitable ice water.
        liquid_water (Quantity):  Precipitable liquid water.
        pwv (Quantity):  Precipitable water vapor.
        humidity (Quantity):  Specific humidity at 10m altitude.
        surface_pressure (Quantity):  Surface pressure.
        surface_temperature (Quantity):  Surface temperature.
        air_temperature (Quantity):  Air temperature at 10m altitude.
        west_wind (Quantity):  Eastward moving wind at 10m altitude.
        south_wind (Quantity):  Northward moving wind at 10m altitude.

    """

    def __init__(
        self,
        time=None,
        ice_water=None,
        liquid_water=None,
        pwv=None,
        humidity=None,
        surface_pressure=None,
        surface_temperature=None,
        air_temperature=None,
        west_wind=None,
        south_wind=None,
    ):
        self._time_val = time
        self._ice_water_val = ice_water
        self._liquid_water_val = liquid_water
        self._pwv_val = pwv
        self._humidity_val = humidity
        self._surface_pressure_val = surface_pressure
        self._surface_temperature_val = surface_temperature
        self._air_temperature_val = air_temperature
        self._west_wind_val = west_wind
        self._south_wind_val = south_wind

    def copy(self):
        return Weather(
            time=self._time_val,
            ice_water=self._ice_water_val,
            liquid_water=self._liquid_water_val,
            pwv=self._pwv_val,
            humidity=self._humidity_val,
            surface_pressure=self._surface_pressure_val,
            surface_temperature=self._surface_temperature_val,
            air_temperature=self._air_temperature_val,
            west_wind=self._west_wind_val,
            south_wind=self._south_wind_val,
        )

    def __eq__(self, other):
        if self._time_val != other._time_val:
            return False
        if not np.isclose(self._ice_water_val.value, other._ice_water_val.value):
            return False
        if not np.isclose(self._liquid_water_val.value, other._liquid_water_val.value):
            return False
        if not np.isclose(self._pwv_val.value, other._pwv_val.value):
            return False
        if not np.isclose(self._humidity_val.value, other._humidity_val.value):
            return False
        if not np.isclose(
            self._surface_pressure_val.value, other._surface_pressure_val.value
        ):
            return False
        if not np.isclose(
            self._surface_temperature_val.value, other._surface_temperature_val.value
        ):
            return False
        if not np.isclose(
            self._air_temperature_val.value, other._air_temperature_val.value
        ):
            return False
        if not np.isclose(self._west_wind_val.value, other._west_wind_val.value):
            return False
        if not np.isclose(self._south_wind_val.value, other._south_wind_val.value):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def _time(self):
        if self._time_val is None:
            raise NotImplementedError("Base class _time() called, but has no value")
        return self._time_val

    @property
    def time(self):
        """The time for these weather properties."""
        return self._time()

    def _ice_water(self):
        if self._ice_water_val is None:
            raise NotImplementedError(
                "Base class _ice_water() called, but has no value"
            )
        return self._ice_water_val

    @property
    def ice_water(self):
        """Total precipitable ice water [kg/m^2] (also [mm])."""
        return self._ice_water()

    def _liquid_water(self):
        if self._liquid_water_val is None:
            raise NotImplementedError(
                "Base class _liquid_water() called, but has no value"
            )
        return self._liquid_water_val

    @property
    def liquid_water(self):
        """Total precipitable liquid water [kg/m^2] (also [mm])."""
        return self._liquid_water()

    def _pwv(self):
        if self._pwv_val is None:
            raise NotImplementedError("Base class _pwv() called, but has no value")
        return self._pwv_val

    @property
    def pwv(self):
        """Total precipitable water vapor [kg/m^2] (also [mm])."""
        return self._pwv()

    def _humidity(self):
        if self._humidity_val is None:
            raise NotImplementedError("Base class _humidity() called, but has no value")
        return self._humidity_val

    @property
    def humidity(self):
        """10-meter specific humidity [kg/kg]"""
        return self._humidity()

    def _surface_pressure(self):
        if self._surface_pressure_val is None:
            raise NotImplementedError(
                "Base class _surface_pressure() called, but has no value"
            )
        return self._surface_pressure_val

    @property
    def surface_pressure(self):
        """Surface pressure [Pa]."""
        return self._surface_pressure()

    def _surface_temperature(self):
        if self._surface_temperature_val is None:
            raise NotImplementedError(
                "Base class _surface_temperature() called, but has no value"
            )
        return self._surface_temperature_val

    @property
    def surface_temperature(self):
        """Surface skin temperature [K]."""
        return self._surface_temperature()

    def _air_temperature(self):
        if self._air_temperature_val is None:
            raise NotImplementedError(
                "Base class _air_temperature() called, but has no value"
            )
        return self._air_temperature_val

    @property
    def air_temperature(self):
        """10-meter air temperature [K]."""
        return self._air_temperature()

    def _west_wind(self):
        if self._west_wind_val is None:
            raise NotImplementedError(
                "Base class _west_wind() called, but has no value"
            )
        return self._west_wind_val

    @property
    def west_wind(self):
        """10-meter eastward wind [m/s]."""
        return self._west_wind()

    def _south_wind(self):
        if self._south_wind_val is None:
            raise NotImplementedError(
                "Base class _south_wind() called, but has no value"
            )
        return self._south_wind_val

    @property
    def south_wind(self):
        """10-meter northward wind [m/s]."""
        return self._south_wind()

    @classmethod
    def _load_hdf5(cls, handle, comm=None, **kwargs):
        """Load base class weather"""
        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        props = dict()

        if handle is not None:
            props["time"] = datetime.datetime.fromtimestamp(
                float(handle.attrs["weather_time"]), tz=timezone.utc
            )
            for attr_name in [
                "ice_water",
                "liquid_water",
                "pwv",
                "humidity",
                "surface_pressure",
                "surface_temperature",
                "air_temperature",
                "west_wind",
                "south_wind",
            ]:
                file_attr = f"weather_{attr_name}"
                props[attr_name] = u.Quantity(handle.attrs[file_attr])
        if need_bcast:
            props = comm.bcast(props, root=0)
        return cls(**props)

    @classmethod
    def load_hdf5(cls, handle, comm=None, **kwargs):
        """Load the weather from an HDF5 group.

        Args:
            handle (h5py.Group):  The group containing the "focalplane" dataset.
            comm (MPI.Comm):  If loading from a file, optional communicator.

        Returns:
            None

        """
        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        weather_class_name = None
        if handle is not None:
            weather_class_name = str(handle.attrs["weather_class"])

        if need_bcast:
            weather_class_name = comm.bcast(weather_class_name, root=0)

        weather_class = import_from_name(weather_class_name)
        return weather_class._load_hdf5(handle, comm=comm, **kwargs)

    def _save_hdf5(self, handle, comm=None, **kwargs):
        handle.attrs["weather_time"] = self.time.astimezone(timezone.utc).timestamp()
        for attr_name in [
            "ice_water",
            "liquid_water",
            "pwv",
            "humidity",
            "surface_pressure",
            "surface_temperature",
            "air_temperature",
            "west_wind",
            "south_wind",
        ]:
            file_attr = f"weather_{attr_name}"
            attr_val = getattr(self, attr_name)
            handle.attrs[file_attr] = str(attr_val)

    def save_hdf5(self, handle, comm=None, **kwargs):
        """Save the weather to an HDF5 group.

        Args:
            handle (h5py.Group):  The parent group for saving site properties.
            comm (MPI.Comm):  If saving to a file, optional communicator.

        Returns:
            None

        """
        if handle is not None:
            handle.attrs["weather_class"] = object_fullname(self.__class__)
        self._save_hdf5(handle, comm=comm, **kwargs)


def read_weather(file):
    """Helper function to read HDF5 format weather file.

    On some filesystems, HDF5 uses a locking mechanism that fails even
    when opening the file readonly.  We work around this by opening
    the file as a binary stream in python and passing that file object
    to h5py.

    Args:
        file (str):  Path to the file.

    Returns:
        (dict):  Weather data dictionary.

    """
    with open(file, "rb") as pf:
        with h5py.File(pf, "r") as hf:
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

    weather_file = resources.files("toast._aux.weather").joinpath(f"{name}.h5")
    with resources.as_file(weather_file) as path:
        if not os.path.isfile(path):
            msg = f"package weather file {path} does not exist"
            raise RuntimeError(msg)
        package_weather_data[name] = read_weather(path)

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
        max_pwv (quantity):  Maximum PWV to draw.
        median_weather (bool):  Instead of random values, return the median ones.

    """

    def __init__(
        self,
        time=None,
        name=None,
        file=None,
        site_uid=0,
        realization=0,
        max_pwv=None,
        median_weather=False,
    ):
        if time is None:
            time = datetime.datetime.now()
        self._name = name
        if self._name is None:
            if file is None:
                # Load a default weather site, useful if we
                # are instantiating the class before loading from
                # HDF5.
                self._data = load_package_weather("atacama")
            else:
                self._data = read_weather(file)
                self._name = file
        else:
            self._data = load_package_weather(self._name)

        self._max_pwv = max_pwv
        if max_pwv is not None:
            self._truncate_distributions("TQV", max_pwv)
        self.median_weather = median_weather

        # Use a separate RNG index for each data type
        self._varindex = {y: x for x, y in enumerate(self._data[0]["data"].keys())}

        self.set(time=time, realization=realization, site_uid=site_uid)

        super().__init__()

    def _truncate_distributions(self, name, max_value):
        # Truncate all distributions, since user may change the time
        for month in range(12):
            prob = self._data[month]["prob"]
            for hour in range(24):
                cdf = self._data[month]["data"][name][hour]
                ind = cdf <= max_value.to_value("mm")
                if np.sum(ind) < 2:
                    raise RuntimeError(f"Cannot truncate {name} to <= {max_value}")
                new_cdf = np.interp(prob, prob[ind] / np.amax(prob[ind]), cdf[ind])
                cdf[:] = new_cdf
        return

    def set(self, time=None, realization=None, site_uid=None):
        """Set new parameters for weather simulation.

        This (re-)sets the time, realization, and site_uid for drawing random weather
        parameters.

        Args:
            time (datetime):  A python date/time in UTC.
            realization (int):  The realization index used for random draw of
                parameters.
            site_uid (int):  The Unique ID for the site, used for random draw of
                parameters.

        Returns:
            None

        """
        if time is not None:
            self._date = time
            self._doy = time.timetuple().tm_yday
            self._year = time.year
            self._hour = time.hour
            # This is the definition of month used in the weather files
            self._month = int((self._doy - 1) // 30.5)
        if realization is not None:
            self._realization = realization
        else:
            self._realization = 0
        if site_uid is not None:
            self._site_uid = site_uid
        else:
            self._site_uid = name_UID(self._name)
        self._draw_values()

    @property
    def name(self):
        """The name of the internal weather object or file."""
        return self._name

    @property
    def realization(self):
        """The current realization."""
        return self._realization

    @property
    def site_uid(self):
        """The current site UID."""
        return self._site_uid

    @property
    def max_pwv(self):
        """The maximum PWV used to truncate the distribution."""
        return self._max_pwv

    def _draw_values(self):
        self._sim_ice_water = self._draw("TQI") * u.mm
        self._sim_liquid_water = self._draw("TQL") * u.mm
        self._sim_pwv = self._draw("TQV") * u.mm
        self._sim_humidity = self._draw("QV10M") * u.mm
        self._sim_surface_pressure = self._draw("PS") * u.Pa
        self._sim_surface_temperature = self._draw("TS") * u.Kelvin
        self._sim_air_temperature = self._draw("T10M") * u.Kelvin
        self._sim_west_wind = self._draw("U10M") * (u.meter / u.second)
        self._sim_south_wind = self._draw("V10M") * (u.meter / u.second)

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

        if self.median_weather:
            x = 0.5
        else:
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
        value = f"<SimWeather : '{self._name}', year = {self._year}, "
        value += f"month = {self._month}, hour = {self._hour}, "
        value += f"site UID = {self._site_uid}, realization = {self._realization}, "
        value += f"median = {self.median_weather})"
        return value

    def __eq__(self, other):
        if self._name != other._name:
            return False
        if self._year != other._year:
            return False
        if self._month != other._month:
            return False
        if self._hour != other._hour:
            return False
        if self._site_uid != other._site_uid:
            return False
        if self._realization != other._realization:
            return False
        if self.median_weather != other.median_weather:
            return False
        if self._max_pwv is None:
            if other._max_pwv is not None:
                return False
        else:
            if other._max_pwv is None:
                return False
            if not np.isclose(self._max_pwv.value, other._max_pwv.value):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def _load_hdf5(cls, handle, comm=None, **kwargs):
        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        weather_name = None
        weather_realization = None
        weather_max_pwv = None
        weather_time = None
        weather_median = None
        site_uid = 0
        if handle is not None:
            weather_name = str(handle.attrs["weather_name"])
            weather_realization = int(handle.attrs["weather_realization"])
            weather_max_pwv = None
            if handle.attrs["weather_max_pwv"] != "NONE":
                weather_max_pwv = u.Quantity(
                    float(handle.attrs["weather_max_pwv"]), u.mm
                )
            weather_time = datetime.datetime.fromtimestamp(
                float(handle.attrs["weather_time"]), tz=timezone.utc
            )
            weather_median = bool(handle.attrs["weather_median"])
            if "site_uid" in handle.attrs:
                site_uid = handle.attrs["site_uid"]

        if need_bcast:
            weather_name = comm.bcast(weather_name, root=0)
            weather_realization = comm.bcast(weather_realization, root=0)
            weather_max_pwv = comm.bcast(weather_max_pwv, root=0)
            weather_time = comm.bcast(weather_time, root=0)
            weather_median = comm.bcast(weather_median, root=0)
            site_uid = comm.bcast(site_uid, root=0)

        return cls(
            time=weather_time,
            name=weather_name,
            site_uid=site_uid,
            realization=weather_realization,
            max_pwv=weather_max_pwv,
            median_weather=weather_median,
        )

    def _save_hdf5(self, handle, comm=None, **kwargs):
        if handle is not None:
            handle.attrs["weather_name"] = str(self.name)
            handle.attrs["weather_realization"] = int(self.realization)
            if self.max_pwv is None:
                handle.attrs["weather_max_pwv"] = "NONE"
            else:
                handle.attrs["weather_max_pwv"] = float(self.max_pwv.to_value(u.mm))
            handle.attrs["weather_time"] = self.time.astimezone(
                timezone.utc
            ).timestamp()
            handle.attrs["weather_median"] = self.median_weather
