# Copyright (c) 2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import astropy.io.fits as pf
import datetime
import numpy as np

from . import rng as rng

from . import timing


class Weather(object):
    """ TOAST Weather objects allow sampling weather parameters.

    The weather parameter distributions are read from site-specific
    TOAST weather files.  The files contain parameter distributions
    for every UTC hour of the day, averaged over months.

    """

    def __init__(self, fname, site=0, realization=0, time=None):
        """ Initialize a weather object

        Args:
            fname(str) : FITS file containing the parameter
                distributions.
            site(int) : Site index for the random number generator.
            realization(int) : Initial realization index, may be
                changed later.

        """
        self._fname = fname
        self.site = site
        self.realization = realization
        if time is None:
            self._time = None
            self._year = None
            self._month = None
            self._hour = None
        else:
            self.set_time(time)
        self._varindex = {}

        hdulist = pf.open(self._fname, 'readonly')

        # Build the probability axis of the cumulative distribution
        # function.  The CDF:s stored for every month, hour and variable
        # all assume the same probability axis.
        prob_start = hdulist[1].header['probstrt']
        prob_stop = hdulist[1].header['probstop']
        nstep = hdulist[1].header['nstep']
        self._prob = np.linspace(prob_start, prob_stop, nstep)

        # Load the CDF:s.  One entry per month.
        self._monthly_cdf = []
        ivar = 0
        for month in range(12):
            self._monthly_cdf.append([])
            hdu = hdulist[1+month]
            # One entry for every hour
            for hour in range(24):
                self._monthly_cdf[month].append({})
            # and one entry for every weather variable:
            #  TQI   : ice water
            #  TQL   : liquid water
            #  TQV   : water vapor
            #  QV10M : specific humidity
            #  PS    : surface pressure
            #  TS    : surface temperature
            #  T10M  : air temperature at 10m
            #  U10M  : eastward wind at 10m
            #  V10M  : northward wind at 10m
            for col in hdu.columns:
                name = col.name
                if name not in self._varindex:
                    self._varindex[name] = ivar
                    ivar += 1
                for hour in range(24):
                    self._monthly_cdf[month][hour][name] \
                        = hdu.data.field(name)[hour]

        hdulist.close()

        self._reset_vars()

        return


    @timing.auto_timer
    def set(self, site, realization, time=None):
        """ Set the weather object state.

        Args:
            site(int) : Site index.
            realization(int) : Realization index.
            time : POSIX timestamp.

        """
        self.site = site
        self.realization = realization
        if time is not None:
            self.set_time(time)
        else:
            self._reset_vars()
        return


    def _reset_vars(self):
        """ Reset the cached random variables.

        """
        self._ice_water = None
        self._liquid_water = None
        self._pwv = None
        self._humidity = None
        self._surface_pressure = None
        self._surface_templerature = None
        self._air_temperature = None
        self._west_wind = None
        self._south_wind = None


    @timing.auto_timer
    def set_time(self, time):
        """ Set the observing time.

        Args:
            time : POSIX timestamp.

        """
        self._time = time
        self._date = datetime.datetime.utcfromtimestamp(self._time)
        self._doy = self._date.timetuple().tm_yday
        self._year = self._date.year
        self._hour = self._date.hour
        # This is the definition of month used in the weather files
        self._month = int((self._doy-1) // 30.5)
        self._reset_vars()
        return


    @timing.auto_timer
    def _draw(self, name):
        """ Return a random parameter value.

        Return a random value for preset variable and time.

        Args:
            name(str): MERRA-2 name for the variable.

        """
        if self._year is None:
            raise RuntimeError('Weather object must be initialized by calling '
                               'set_time(time)')
        # Set the RNG counters for this variable and time
        counter1 = self._varindex[name]
        counter2 = (self._year*366 + self._doy)*24 + self._hour
        # Get a uniform random number for inverse sampling
        x = rng.random(
            1, sampler="uniform_01", key=(self.site, self.realization),
            counter=(counter1, counter2))[0]
        # Sample the variable from the inverse cumulative distribution function
        cdf = self._monthly_cdf[self._month][self._hour][name]

        return np.interp(x, self._prob, cdf)


    @property
    @timing.auto_timer
    def ice_water(self):
        """ Total precipitable ice water [kg/m^2] (also [mm]).

        Ice water column at the observing site at the preset time and
        for the preset realization.

        """
        if self._ice_water is None:
            self._ice_water = self._draw('TQI')
        return self._ice_water


    @property
    @timing.auto_timer
    def liquid_water(self):
        """ Total precipitable liquid water [kg/m^2] (also [mm]).

        Liquid water column at the observing site at the preset time and
        for the preset realization.

        """
        if self._liquid_water is None:
            self._liquid_water = self._draw('TQL')
        return self._liquid_water


    @property
    @timing.auto_timer
    def pwv(self):
        """ Total precipitable water vapor [kg/m^2] (also [mm]).

        Water vapor column at the observing site at the preset time and
        for the preset realization.

        """
        if self._pwv is None:
            self._pwv = self._draw('TQV')
        return self._pwv


    @property
    @timing.auto_timer
    def humidity(self):
        """ 10-meter specific humidity [kg/kg]

        Water vapor concentration at the observing site 10 meters above
        ground at the preset time and for the preset realization.

        """
        if self._humidity is None:
            self._humidity = self._draw('QV10M')
        return self._humidity


    @property
    @timing.auto_timer
    def surface_pressure(self):
        """ Surface pressure [Pa].

        Surface at the observing site at the preset time and for the
        preset realization.

        """
        if self._surface_pressure is None:
            self._surface_pressure = self._draw('PS')
        return self._surface_pressure


    @property
    @timing.auto_timer
    def surface_temperature(self):
        """ Surface skin temperature [K].

        Surface temperature at the observing site at the preset time and
        for the preset realization.

        """
        if self._surface_temperature is None:
            self._surface_temperature = self._draw('TS')
        return self._surface_temperature


    @property
    @timing.auto_timer
    def air_temperature(self):
        """ 10-meter air temperature [K].

        Air temperature at the observing site 10 meters above ground
        at the preset time and for the preset realization.

        """
        if self._air_temperature is None:
            self._air_temperature = self._draw('T10M')
        return self._air_temperature


    @property
    @timing.auto_timer
    def west_wind(self):
        """ 10-meter eastward wind [m/s].

        Eastward wind at the observing site 10 meters above ground
        at the preset time and for the preset realization.

        """
        if self._west_wind is None:
            self._west_wind = self._draw('U10M')
        return self._west_wind


    @property
    @timing.auto_timer
    def south_wind(self):
        """ 10-meter northward wind [m/s].

        Northward wind at the observing site 10 meters above ground
        at the preset time and for the preset realization.

        """
        if self._south_wind is None:
            self._south_wind = self._draw('V10M')
        return self._south_wind
