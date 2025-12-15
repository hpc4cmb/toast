# Copyright (c) 2019-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Classes for describing the instrument model.

Note that these classes usually are stored under the Observation.telescope
object.  Since these must be instantiated before creating an observation,
The interface to the load_hdf5 and save_hdf5 methods are different than
for objects designed to be saved / loaded with a pre-existing Observation
instance.

"""

import datetime
import os
import sys
from datetime import timezone

import astropy.coordinates as coord
import astropy.time as astime
import h5py
import numpy as np
from astropy import units as u
from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
from astropy.table import Column, QTable
from scipy.constants import c, h, k

try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson

import tomlkit

from . import qarray
from . import qarray as qa
from ._libtoast import integrate_simpson
from .noise_sim import AnalyticNoise
from .timing import Timer, function_timer
from .utils import (
    Environment,
    Logger,
    hdf5_use_serial,
    name_UID,
    object_fullname,
    table_write_parallel_hdf5,
    import_from_name,
)

# CMB temperature
TCMB = 2.72548


class Site(object):
    """Site base class.

    Args:
        name (str):  Site name
        uid (int):  Unique identifier.  If not specified, constructed from a hash
            of the site name.

    """

    def __init__(self, name, uid=None):
        self.name = name
        self.uid = uid
        if self.uid is None:
            self.uid = name_UID(self.name)

    def _position(self, times):
        raise NotImplementedError("Derived class must implement _position()")

    def position(self, times):
        """Get the site position in solar system barycentric cartesian vectors.

        Given timestamps in POSIX seconds since 1970 (UTC), return the position as
        solar system coordinates.

        Args:
            times (array):  The timestamps.

        Returns:
            (array):  The position vectors.

        """
        return self._position(times)

    def _velocity(self, times):
        raise NotImplementedError("Derived class must implement _velocity()")

    def velocity(self, times):
        """Get the site velocity in solar system barycentric cartesian vectors.

        Given timestamps in POSIX seconds since 1970 (UTC), return the velocity as
        quaternions in solar system barycentric coordinates.

        Args:
            times (array):  The timestamps.

        Returns:
            (array):  The velocity vectors.

        """
        return self._velocity(times)

    def position_velocity(self, times):
        """Get the site position and velocity.

        Convenience function to simultaneously return the position and velocity.

        Args:
            times (array):  The timestamps.

        Returns:
            (tuple):  The position and velocity arrays of vectors.

        """
        if hasattr(self, "_position_velocity"):
            return self._position_velocity(times)
        else:
            p = self._position(times)
            v = self._velocity(times)
        return (p, v)

    def __repr__(self):
        value = "<Site '{}' : uid = {}>".format(self.name, self.uid)
        return value

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.uid != other.uid:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def _load_hdf5(cls, handle, comm=None, **kwargs):
        raise NotImplementedError("Derived class must implement _load_hdf5()")

    @classmethod
    def load_hdf5(cls, handle, comm=None, **kwargs):
        """Load the site from an HDF5 group.

        This checks for the site_class attribute and dispatches to the correct
        class method.

        Args:
            handle (h5py.Group):  The group containing the site information.
            comm (MPI.Comm):  If loading from a file, optional communicator.

        Returns:
            None

        """
        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        site_class_name = None
        if handle is not None:
            site_class_name = str(handle.attrs["site_class"])
        if need_bcast:
            site_class_name = comm.bcast(site_class_name, root=0)
        site_class = import_from_name(site_class_name)
        return site_class._load_hdf5(handle, comm=comm, **kwargs)

    def _save_hdf5(self, handle, comm=None, **kwargs):
        raise NotImplementedError("Derived class must implement _save_hdf5()")

    def save_hdf5(self, handle, comm=None, **kwargs):
        """Save the site to an HDF5 group.

        Args:
            handle (h5py.Group):  The parent group for saving site properties.
            comm (MPI.Comm):  If saving to a file, optional communicator.

        Returns:
            None

        """
        if handle is not None:
            handle.attrs["site_class"] = object_fullname(self.__class__)
        self._save_hdf5(handle, comm=comm, **kwargs)


class GroundSite(Site):
    """Site on the Earth.

    This represents a fixed location on the Earth.

    Args:
        name (str):  Site name
        lat (Quantity):  Site latitude.
        lon (Quantity):  Site longitude.
        alt (Quantity):  Site altitude.
        uid (int):  Unique identifier.  If not specified, constructed from a hash
            of the site name.
        weather (Weather):  Weather information for this site.
    """

    def __init__(self, name, lat, lon, alt, uid=None, weather=None):
        super().__init__(name, uid)
        self.earthloc = coord.EarthLocation.from_geodetic(lon, lat, height=alt)
        self.weather = weather

    def __repr__(self):
        value = "<GroundSite '{}' : uid = {}, lon = {}, lat = {}, alt = {}, weather = {}>".format(
            self.name,
            self.uid,
            self.earthloc.lon,
            self.earthloc.lat,
            self.earthloc.height,
            self.weather,
        )
        return value

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.uid != other.uid:
            return False
        if not np.isclose(other.earthloc.lon, self.earthloc.lon):
            return False
        if not np.isclose(other.earthloc.lat, self.earthloc.lat):
            return False
        if not np.isclose(other.earthloc.height, self.earthloc.height):
            return False
        if self.weather != other.weather:
            return False
        return True

    def _position_velocity(self, times):
        # Compute data at 10 second intervals and interpolate.  If the timestamps are
        # more coarsely sampled than that, just compute those times directly.
        sparse_incr = 10.0
        do_interp = True
        if len(times) < 100 or (times[1] - times[0]) > sparse_incr:
            do_interp = False

        if do_interp:
            n_sparse = int((times[-1] - times[0]) / sparse_incr)
            sparse_times = np.linspace(times[0], times[-1], num=n_sparse, endpoint=True)
        else:
            n_sparse = len(times)
            sparse_times = times
        pos_x = np.zeros(n_sparse, np.float64)
        pos_y = np.zeros(n_sparse, np.float64)
        pos_z = np.zeros(n_sparse, np.float64)
        vel_x = np.zeros(n_sparse, np.float64)
        vel_y = np.zeros(n_sparse, np.float64)
        vel_z = np.zeros(n_sparse, np.float64)
        for i, t in enumerate(sparse_times):
            atime = astime.Time(t, format="unix")
            p, v = coord.get_body_barycentric_posvel("earth", atime)
            # FIXME:  apply translation from earth center to earth location.
            # itrs = self.earthloc.get_itrs(obstime)
            pm = p.xyz.to_value(u.kilometer)
            vm = v.xyz.to_value(u.kilometer / u.second)
            pos_x[i] = pm[0]
            pos_y[i] = pm[1]
            pos_z[i] = pm[2]
            vel_x[i] = vm[0]
            vel_y[i] = vm[1]
            vel_z[i] = vm[2]

        if do_interp:
            pos_x = np.interp(times, sparse_times, pos_x)
            pos_y = np.interp(times, sparse_times, pos_y)
            pos_z = np.interp(times, sparse_times, pos_z)
            vel_x = np.interp(times, sparse_times, vel_x)
            vel_y = np.interp(times, sparse_times, vel_y)
            vel_z = np.interp(times, sparse_times, vel_z)
        pos = np.stack([pos_x, pos_y, pos_z], axis=-1)
        vel = np.stack([vel_x, vel_y, vel_z], axis=-1)
        return pos, vel

    def _position(self, times):
        p, v = self._position_velocity(times)
        return p

    def _velocity(self, times):
        p, v = self._position_velocity(times)
        return v

    @classmethod
    def _load_hdf5(cls, handle, comm=None, **kwargs):
        """Load the site from an HDF5 group.

        Args:
            handle (h5py.Group):  The group containing the site information.
            comm (MPI.Comm):  If loading from a file, optional communicator.

        Returns:
            None

        """
        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        site_name = None
        site_uid = None
        site_alt_m = None
        site_lat_deg = None
        site_lon_deg = None
        weather_class_name = None
        if handle is not None:
            site_name = str(handle.attrs["site_name"])
            site_uid = int(handle.attrs["site_uid"])
            site_alt_m = float(handle.attrs["site_alt_m"])
            site_lat_deg = float(handle.attrs["site_lat_deg"])
            site_lon_deg = float(handle.attrs["site_lon_deg"])
            if "weather_class" in handle.attrs:
                weather_class_name = str(handle.attrs["weather_class"])

        if need_bcast:
            site_name = comm.bcast(site_name, root=0)
            site_uid = comm.bcast(site_uid, root=0)
            site_alt_m = comm.bcast(site_alt_m, root=0)
            site_lat_deg = comm.bcast(site_lat_deg, root=0)
            site_lon_deg = comm.bcast(site_lon_deg, root=0)
            weather_class_name = comm.bcast(weather_class_name, root=0)

        weather = None
        if weather_class_name is not None:
            weather_class = import_from_name(weather_class_name)
            weather = weather_class.load_hdf5(handle, comm=comm, **kwargs)

        return cls(
            site_name,
            site_lat_deg * u.degree,
            site_lon_deg * u.degree,
            site_alt_m * u.meter,
            uid=site_uid,
            weather=weather,
        )

    def _save_hdf5(self, handle, comm=None, **kwargs):
        if handle is not None:
            handle.attrs["site_name"] = self.name
            handle.attrs["site_uid"] = self.uid
            handle.attrs["site_lat_deg"] = float(self.earthloc.lat.to_value(u.degree))
            handle.attrs["site_lon_deg"] = float(self.earthloc.lon.to_value(u.degree))
            handle.attrs["site_alt_m"] = float(self.earthloc.height.to_value(u.meter))
            if self.weather is not None:
                self.weather.save_hdf5(handle, comm=comm, **kwargs)


class SpaceSite(Site):
    """Site with no atmosphere.

    This represents a location beyond the Earth's atmosphere.  In practice, this
    should be sub-classed for real satellite experiments.

    Args:
        name (str):  Site name
        uid (int):  Unique identifier.  If not specified, constructed from a hash
            of the site name.

    """

    def __init__(self, name, uid=None):
        super().__init__(name, uid)

    def __repr__(self):
        value = "<SpaceSite '{}' : uid = {}>".format(self.name, self.uid)
        return value

    def _position_velocity(self, times):
        # Compute data at 10 minute intervals and interpolate.  If the timestamps are
        # more coarsely sampled than that, just compute those times directly.
        sparse_incr = 600.0
        do_interp = True
        if len(times) < 100 or (times[1] - times[0]) > sparse_incr:
            do_interp = False

        if do_interp:
            n_sparse = 1 + int((times[-1] - times[0]) / sparse_incr)
            sparse_times = np.linspace(times[0], times[-1], num=n_sparse, endpoint=True)
        else:
            n_sparse = len(times)
            sparse_times = times
        pos_x = np.zeros(n_sparse, np.float64)
        pos_y = np.zeros(n_sparse, np.float64)
        pos_z = np.zeros(n_sparse, np.float64)
        vel_x = np.zeros(n_sparse, np.float64)
        vel_y = np.zeros(n_sparse, np.float64)
        vel_z = np.zeros(n_sparse, np.float64)
        for i, t in enumerate(sparse_times):
            atime = astime.Time(t, format="unix")
            # Get the satellite position and velocity in the equatorial frame (ICRS)
            p, v = coord.get_body_barycentric_posvel("earth", atime)
            # FIXME:  apply translation from earth center to L2.
            pm = p.xyz.to_value(u.kilometer)
            vm = v.xyz.to_value(u.kilometer / u.second)
            pos_x[i] = pm[0]
            pos_y[i] = pm[1]
            pos_z[i] = pm[2]
            vel_x[i] = vm[0]
            vel_y[i] = vm[1]
            vel_z[i] = vm[2]

        if do_interp:
            pos_x = np.interp(times, sparse_times, pos_x)
            pos_y = np.interp(times, sparse_times, pos_y)
            pos_z = np.interp(times, sparse_times, pos_z)
            vel_x = np.interp(times, sparse_times, vel_x)
            vel_y = np.interp(times, sparse_times, vel_y)
            vel_z = np.interp(times, sparse_times, vel_z)
        pos = np.stack([pos_x, pos_y, pos_z], axis=-1)
        vel = np.stack([vel_x, vel_y, vel_z], axis=-1)
        return pos, vel

    def _position(self, times):
        p, v = self._position_velocity(times)
        return p

    def _velocity(self, times):
        p, v = self._position_velocity(times)
        return v

    @classmethod
    def _load_hdf5(cls, handle, comm=None, **kwargs):
        """Load the site from an HDF5 group.

        Args:
            handle (h5py.Group):  The group containing the site information.
            comm (MPI.Comm):  If loading from a file, optional communicator.

        Returns:
            None

        """
        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        site_name = None
        site_uid = None
        if handle is not None:
            site_name = str(handle.attrs["site_name"])
            site_uid = int(handle.attrs["site_uid"])

        if need_bcast:
            site_name = comm.bcast(site_name, root=0)
            site_uid = comm.bcast(site_uid, root=0)

        return cls(site_name, uid=site_uid)

    def _save_hdf5(self, handle, comm=None, **kwargs):
        if handle is not None:
            handle.attrs["site_name"] = self.name
            handle.attrs["site_uid"] = self.uid


class Bandpass(object):
    """Class that contains the bandpass information for an entire focalplane."""

    @function_timer
    def __init__(self, bandcenters, bandwidths, nstep=101):
        """All units in GHz

        Args :
            bandcenters(dict) : Dictionary of bandpass centers
            bandwidths(dict) : Dictionary of bandpass widths
            nstep(int) : Number of interplation steps to use in `convolve()`
        """
        self.nstep = nstep
        self.dets = []
        self._fmin = {}
        self._fmax = {}
        for name in bandcenters:
            self.dets.append(name)
            center = bandcenters[name]
            width = bandwidths[name]
            self._fmin[name] = center - 0.5 * width
            self._fmax[name] = center + 0.5 * width
        # The interpolated bandpasses will be cached as needed
        self._fmin_tot = None
        self._fmax_tot = None
        self._freqs = {}
        self._bandpass = {}
        self._kcmb2jysr = {}
        self._kcmb2krj = {}
        self._kcmb2w = {}

    @function_timer
    def get_range(self, det=None):
        """Return the maximum range of frequencies needed for convolution."""
        if det is not None:
            return self._fmin[det], self._fmax[det]
        elif self._fmin_tot is None:
            self._fmin_tot = min(self._fmin.values())
            self._fmax_tot = max(self._fmax.values())
        return self._fmin_tot, self._fmax_tot

    @function_timer
    def center_frequency(self, det, alpha=-1):
        """Return the effective central frequency for a given spectral index"""

        # Which delta function bandpass would produce the same flux density
        freqs = self.freqs(det)
        if alpha == 0:
            # The equation is singular at alpha == 0. Evaluate it on both sides
            # and return the average
            delta = 1e-6
            alpha1 = alpha - delta
            eff1 = self.convolve(det, freqs, freqs.to_value(u.Hz) ** alpha1) ** (
                1 / alpha1
            )
            alpha2 = alpha + delta
            eff2 = self.convolve(det, freqs, freqs.to_value(u.Hz) ** alpha2) ** (
                1 / alpha2
            )
            eff = 0.5 * (eff1 + eff2)
        else:
            # Very simple closed form
            eff = self.convolve(det, freqs, freqs.to_value(u.Hz) ** alpha) ** (
                1 / alpha
            )

        return eff * u.Hz

    @function_timer
    def optical_loading(self, det, T):
        """Return the optical loading of a blackbody source.
        Assumes a diffraction-limited, single-moded polarimeter
        and perfect optical efficiency
        arXiv:1806.04316

        Args:
            det(str) : detector name
            T(float) : Source temperature in Kelvin

        Returns:
            (float) : The optical loading in Watts
        """

        bandpass = self.bandpass(det)
        # Normalize the bandpass to peak at 1
        bandpass = bandpass / np.amax(bandpass)
        freqs = self.freqs(det).to_value(u.Hz)

        # Power spectral density
        S = h * freqs / (np.exp(h * freqs / k / T) - 1)

        # Integrate over frequency to get power
        power = integrate_simpson(freqs, S * bandpass)

        return power

    @function_timer
    def _get_unit_conversion_coefficients(self, det):
        """Compute and cache the unit conversion coefficients for one detector"""

        if (
            det not in self._kcmb2jysr
            or det not in self._kcmb2krj
            or det not in self._kcmb2w
        ):
            # The calculation is a copy from the Hildebrandt and Macias-Perez IDL module for Planck

            nu_cmb = k * TCMB / h
            alpha = 2 * k**3 * TCMB**2 / h**2 / c**2

            cfreq = self.center_frequency(det).to_value(u.Hz)
            freqs = self.freqs(det).to_value(u.Hz)
            bandpass = self.bandpass(det)

            x = freqs / nu_cmb
            db_dt = alpha * x**4 * np.exp(x) / (np.exp(x) - 1) ** 2
            db_dt_rj = 2 * freqs**2 * k / c**2

            self._kcmb2jysr[det] = (
                1e26
                * integrate_simpson(freqs, db_dt * bandpass)
                / integrate_simpson(freqs, cfreq / freqs * bandpass)
            )
            self._kcmb2krj[det] = integrate_simpson(
                freqs, db_dt * bandpass
            ) / integrate_simpson(freqs, db_dt_rj * bandpass)

            # K_CMB->W conversion is from the BoloCalc paper, arXiv:1806.04316
            bandpass = bandpass / np.amax(bandpass)
            self._kcmb2w[det] = integrate_simpson(
                freqs,
                k * (x / (np.exp(x) - 1)) ** 2 * np.exp(x) * bandpass,
            )

        return

    @function_timer
    def freqs(self, det):
        if det not in self._freqs:
            fmin = self._fmin[det].to_value(u.Hz)
            fmax = self._fmax[det].to_value(u.Hz)
            self._freqs[det] = np.linspace(fmin, fmax, self.nstep) * u.Hz
        return self._freqs[det]

    @function_timer
    def bandpass(self, det):
        if det not in self._bandpass:
            # Normalize and interpolate the bandpass
            freqs = self.freqs(det)
            try:
                # If we have a tabulated bandpass, interpolate it
                self._bandpass[det] = np.interp(
                    freqs.to_value(u.Hz),
                    self._bins[det].to_value(u.Hz),
                    self._values[det],
                )
            except AttributeError:
                self._bandpass[det] = np.ones(self.nstep)

            # norm = simpson(self.bandpass[det], x=self.freqs[det])
            norm = integrate_simpson(freqs.to_value(u.Hz), self._bandpass[det])
            if norm == 0:
                raise RuntimeError("Bandpass cannot be normalized")
            self._bandpass[det] /= norm

        return self._bandpass[det]

    @function_timer
    def kcmb2jysr(self, det):
        """Return the unit conversion between K_CMB and Jy/sr"""
        self._get_unit_conversion_coefficients(det)
        return self._kcmb2jysr[det]

    @function_timer
    def kcmb2krj(self, det):
        """Return the unit conversion between K_CMB and K_RJ"""
        self._get_unit_conversion_coefficients(det)
        return self._kcmb2krj[det]

    @function_timer
    def kcmb2w(self, det):
        """Return the unit conversion between K_CMB and W"""
        self._get_unit_conversion_coefficients(det)
        return self._kcmb2w[det]

    @function_timer
    def convolve(self, det, freqs, spectrum, rj=False):
        """Convolve the provided spectrum with the detector bandpass

        Args:
            det(str):  Detector name
            freqs(array of floats):  Spectral bin locations
            spectrum(array of floats):  Spectral bin values
            rj(bool):  Input spectrum is in Rayleigh-Jeans units and
                should be converted into thermal units for convolution

        Returns:
            (array):  The bandpass-convolved spectrum
        """
        freqs_det = self.freqs(det)
        bandpass_det = self.bandpass(det)

        # Interpolate spectrum values to bandpass frequencies
        spectrum_det = np.interp(
            freqs_det.to_value(u.Hz), freqs.to_value(u.Hz), spectrum
        )

        if rj:
            # From brightness to thermodynamic units
            x = h * freqs_det.to_value(u.Hz) / k / TCMB
            rj2cmb = (x / (np.exp(x / 2) - np.exp(-x / 2))) ** -2
            spectrum_det *= rj2cmb

        # Average across the bandpass
        convolved = integrate_simpson(
            freqs_det.to_value(u.Hz), spectrum_det * bandpass_det
        )

        return convolved


class Focalplane(object):
    """Class representing the focalplane for one observation.

    The detector_data Table may store arbitrary columns, but several are required.
    They include:

        "name":  The detector name.
        "quat":  Each row should be a 4-element numpy array.
        "gamma":  If using a half wave plate, we need the rotation angle of the
            detector polarization orientation from the focalplane frame X-axis.

    Some columns are optional:

        "uid":  Unique integer ID for each detector.  Computed from detector name if
            not specified.
        "pol_angle":  Quantity to specify the polarization angle.  Default assumes
            the polarization sensitive direction is aligned with the detector
            quaternion rotation.  Computed if not specified.
        "pol_leakage":  Float value "epsilon" between 0-1.  Set to zero by default.
        "pol_efficiency":  Float value "eta" = (1 - epsilon) / (1 + epsilon).  Set
            to one by default.
        "fwhm":  Quantity with the nominal beam FWHM.  Used for plotting and for
            smoothing of simulated sky signal with PySM.
        "bandcenter":  Quantity for the band center.  Used for bandpass integration
            with PySM simulations.
        "bandwidth":  Quantity for width of the band.  Used for bandpass integration
            with PySM simulations.
        "psd_net":  The detector sensitivity.  Quantity used to create a synthetic
            noise model with the DefaultNoiseModel operator.
        "psd_fknee":  Quantity used to create a synthetic noise model with the
            DefaultNoiseModel operator.
        "psd_fmin":  Quantity used to create a synthetic noise model with the
            DefaultNoiseModel operator.
        "psd_alpha":  Quantity used to create a synthetic noise model with the
            DefaultNoiseModel operator.
        "elevation_noise_a" and "elevation_noise_c":  Parameters of elevation scaling
            noise model: PSD_{out} = PSD_{ref} * (a / sin(el) + c)^2.  Only applicable
            to ground data.
        "pwv_a0", "pwv_a1" and "pwv_a2":  quadratic fit of the NET modulation by
            PWV.  Only applicable to ground data.

    Args:
        detector_data (QTable):  Table of detector properties.
        field_of_view (Quantity):  Angular diameter of the focal plane.  Used to
            increase the effective size of the focalplane when simulating atmosphere,
            etc.  Will be calculated from the detector offsets by default.
        sample_rate (Quantity):  The common (nominal) sample rate for all detectors.
        thinfp (int):  Only sample the detectors in the file.

    """

    XAXIS, YAXIS, ZAXIS = np.eye(3)

    @function_timer
    def __init__(
        self,
        detector_data=None,
        field_of_view=None,
        sample_rate=None,
        thinfp=None,
    ):
        self.detector_data = detector_data
        self.field_of_view = field_of_view
        self.sample_rate = sample_rate
        self.thinfp = thinfp
        if detector_data is not None and len(detector_data) > 0:
            # We have some dets
            self._initialize()

    @function_timer
    def _initialize(self):
        log = Logger.get()

        if self.thinfp is not None:
            # Pick only every `thinfp` pixel on the focal plane
            ndet = len(self.detector_data)
            for idet in range(ndet - 1, -1, -1):
                if int(idet // 2) % self.thinfp != 0:
                    del self.detector_data[idet]

        # Add UID if not given
        if "uid" not in self.detector_data.colnames:
            self.detector_data.add_column(
                Column(
                    name="uid", data=[name_UID(x["name"]) for x in self.detector_data]
                )
            )

        # Build index of detector to table row
        self._det_to_row = {y["name"]: x for x, y in enumerate(self.detector_data)}

        if self.field_of_view is None:
            self._compute_fov()
        self._get_pol_angles()
        self._get_pol_efficiency()
        self._get_bandpass()

    @function_timer
    def _get_bandpass(self):
        """Use the bandpass parameters to instantiate a bandpass model"""

        if "bandcenter" in self.detector_data.colnames:
            bandcenter = {}
            bandwidth = {}
            for row in self.detector_data:
                name = row["name"]
                bandcenter[name] = row["bandcenter"]
                bandwidth[name] = row["bandwidth"]
            self.bandpass = Bandpass(bandcenter, bandwidth)
        else:
            self.bandpass = None
        return

    @function_timer
    def _compute_fov(self):
        """Compute the field of view"""
        # Find the largest distance from the bore sight
        cosangs = list()
        for row in self.detector_data:
            quat = row["quat"]
            vec = qarray.rotate(quat, self.ZAXIS)
            cosangs.append(np.dot(self.ZAXIS, vec))
        mincos = np.amin(cosangs)
        # Add a very small margin to avoid numeric issues
        # in the atmospheric simulation
        self.field_of_view = 1.01 * 2.0 * np.arccos(mincos) * u.radian
        # If we just have boresight detectors, we will need to give this some non-zero
        # value.
        if self.field_of_view == 0:
            self.field_of_view = 1.0 * u.degree

    @function_timer
    def _get_pol_angles(self):
        """Get the detector polarization angles from the quaternions"""

        if "pol_angle" not in self.detector_data.colnames:
            n_rows = len(self.detector_data)
            self.detector_data.add_column(
                Column(name="pol_angle", length=n_rows, unit=u.radian)
            )
            for row in self.detector_data:
                quat = row["quat"]
                a = quat[3]
                d = quat[2]
                pol_angle = np.arctan2(2 * a * d, a**2 - d**2) % np.pi
                row["pol_angle"] = pol_angle * u.radian

    @function_timer
    def _get_pol_efficiency(self):
        """Get the polarization efficiency from polarization leakage or vice versa"""

        n_rows = len(self.detector_data)
        if ("pol_leakage" in self.detector_data.colnames) and (
            "pol_efficiency" in self.detector_data.colnames
        ):
            # Check that efficiency and leakage are consistent
            epsilon = self.detector_data["pol_leakage"]
            eta = self.detector_data["pol_efficiency"]
            np.testing.assert_allclose(
                eta,
                (1 + epsilon) / (1 - epsilon),
                rtol=1e-6,
                err_msg="inconsistent polarization leakage and efficiency",
            )
            return
        elif "pol_leakage" in self.detector_data.colnames:
            self.detector_data.add_column(
                Column(
                    name="pol_efficiency",
                    data=[(1 - x) / (1 + x) for x in self.detector_data["pol_leakage"]],
                )
            )
        elif "pol_efficiency" in self.detector_data.colnames:
            self.detector_data.add_column(
                Column(
                    name="pol_leakage",
                    data=[
                        (1 - x) / (1 + x) for x in self.detector_data["pol_efficiency"]
                    ],
                )
            )
        else:
            self.detector_data.add_column(
                Column(name="pol_efficiency", data=np.ones(n_rows))
            )
            self.detector_data.add_column(
                Column(name="pol_leakage", data=np.zeros(n_rows))
            )

    def __contains__(self, key):
        return key in self._det_to_row

    def __getitem__(self, key):
        return self.detector_data[self._det_to_row[key]]

    def __setitem__(self, key, value):
        if key not in self._det_to_row:
            msg = "cannot assign to non-existent detector '{}'".format(key)
            raise ValueError(msg)
        indx = self._det_to_row[key]
        if hasattr(value, "fields"):
            # numpy structured array
            if value.fields is None:
                raise ValueError("assignment value must be structured")
            for cname, ctype in value.fields.items():
                if cname not in self.detector_data.colnames:
                    msg = "assignment value element '{}' is not a det column".format(
                        cname
                    )
                    raise ValueError(msg)
                self.detector_data[indx][cname] = value[cname]
        elif hasattr(value, "colnames"):
            # table row
            for c in value.colnames:
                if c not in self.detector_data.colnames:
                    msg = "assignment value element '{}' is not a det column".format(c)
                    raise ValueError(msg)
                self.detector_data[indx][c] = value[c]
        else:
            # see if it is like a dictionary
            try:
                for k, v in value.items():
                    if k not in self.detector_data.colnames:
                        msg = (
                            "assignment value element '{}' is not a det column".format(
                                k
                            )
                        )
                        raise ValueError(msg)
                    self.detector_data[indx][k] = v
            except Exception:
                raise ValueError(
                    "assignment value must be a dictionary, Row, or structured array"
                )

    @property
    def detectors(self):
        return list(self._det_to_row.keys())

    @property
    def properties(self):
        return list(self.detector_data.colnames)

    @property
    def n_detectors(self):
        return len(self._det_to_row.keys())

    def keys(self):
        return self.detectors

    @function_timer
    def detector_groups(self, column):
        """Group detectors by a common value in one property.

        This returns a dictionary whose keys are the unique values of the specified
        detector_data column.  The values for each key are a list of detectors that
        have that value.  This can be useful for creating detector sets for data
        distribution or for considering detectors with correlations.

        Since the column values will be used for dictionary keys, the column must
        be a data type which is hashable.

        Args:
            column (str):  The detector_data column.

        Returns:
            (dict):  The detector names grouped by unique column values.

        """
        if column not in self.detector_data.colnames:
            raise RuntimeError(f"'{column}' is not a valid det data column")
        detgroups = dict()
        for d in self.detectors:
            indx = self._det_to_row[d]
            val = self.detector_data[column][indx]
            if val not in detgroups:
                detgroups[val] = list()
            detgroups[val].append(d)
        return detgroups

    def __repr__(self):
        value = "<Focalplane: {} detectors, sample_rate = {} Hz, FOV = {} deg, detectors = [".format(
            len(self.detector_data),
            self.sample_rate.to_value(u.Hz),
            self.field_of_view.to_value(u.degree),
        )
        value += "{} .. {}".format(self.detectors[0], self.detectors[-1])
        value += "]>"
        return value

    def __eq__(self, other):
        if self.sample_rate != other.sample_rate:
            return False
        if self.field_of_view != other.field_of_view:
            return False
        if self.detectors != other.detectors:
            return False
        if self.detector_data.colnames != other.detector_data.colnames:
            return False
        if not self.detector_data.values_equal(other.detector_data):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def _load_hdf5(
        cls, handle, comm=None, detectors=None, file_det_sets=None, **kwargs
    ):
        """Load a base class Focalplane"""
        log = Logger.get()

        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        detector_data = None
        sample_rate = None
        field_of_view = None
        if handle is not None:
            detector_data = read_table_hdf5(handle, path="focalplane")
            sample_rate = detector_data.meta["sample_rate"]
            field_of_view = detector_data.meta["field_of_view"]

        if need_bcast:
            detector_data = comm.bcast(detector_data, root=0)
            sample_rate = comm.bcast(sample_rate, root=0)
            field_of_view = comm.bcast(field_of_view, root=0)

        log.debug_rank(
            f"Focalplane has {len(detector_data)} detectors that span "
            f"{field_of_view.to_value(u.deg):.3f} degrees and are sampled at "
            f"{sample_rate.to_value(u.Hz)} Hz.",
            comm=comm,
        )

        return cls(
            detector_data=detector_data,
            field_of_view=field_of_view,
            sample_rate=sample_rate,
            **kwargs,
        )

    @classmethod
    def load_hdf5(cls, handle, comm=None, **kwargs):
        """Load the focalplane from an HDF5 group.

        Args:
            handle (h5py.Group):  The group containing the "focalplane" dataset.
            comm (MPI.Comm):  If loading from a file, optional communicator.

        Returns:
            None

        """
        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        focalplane_class_name = None
        if handle is not None:
            focalplane_class_name = handle.attrs["focalplane_class"]
        if need_bcast:
            focalplane_class_name = comm.bcast(focalplane_class_name, root=0)

        focalplane_class = import_from_name(focalplane_class_name)
        return focalplane_class._load_hdf5(handle, comm=comm, **kwargs)

    def _save_hdf5(self, handle, comm=None, **kwargs):
        self.detector_data.meta["sample_rate"] = self.sample_rate
        self.detector_data.meta["field_of_view"] = self.field_of_view
        table_write_parallel_hdf5(self.detector_data, handle, "focalplane", comm=comm)

    def save_hdf5(self, handle, comm=None, **kwargs):
        """Save the focalplane to an HDF5 group.

        Args:
            handle (h5py.Group):  The parent group of the focalplane dataset.
            comm (MPI.Comm):  If saving to a file, optional communicator.

        Returns:
            None

        """
        if handle is not None:
            handle.attrs["focalplane_class"] = object_fullname(self.__class__)
        self._save_hdf5(handle, comm=comm, **kwargs)


class Session(object):
    """Class representing an observing session.

    A session consists of multiple Observation instances with different sets of
    detectors and possibly different sample rates / times.  However these
    observations are on the same physical telescope and over the same broad
    time range.  A session simply tracks that time range and a unique ID which
    can be used to group the relevant observations.

    Args:
        name (str):  The name of the session.
        uid (int):  The Unique ID of the session.  If not specified, it will be
            constructed from a hash of the name and the optional start/stop times.
        start (datetime):  The overall start of the session.
        end (datetime):  The overall end of the session.

    """

    def __init__(self, name, uid=None, start=None, end=None):
        self.name = name
        for t in start, end:
            if t is not None and not isinstance(t, datetime.datetime):
                raise RuntimeError("Session start/end must be a datetime or None")
        if uid is not None:
            self.uid = uid
        else:
            # Append start and end times to the session name before
            # evaluating the hash.  This reduces the risk of clashing UIDs.
            session_name = name
            if start is not None:
                session_name += start.ctime()
            if end is not None:
                session_name += end.ctime()
            self.uid = name_UID(session_name)
        self.start = start
        self.end = end

    def __repr__(self):
        value = "<Session '{}': uid = {}, start = {}, end = {}".format(
            self.name, self.uid, self.start, self.end
        )
        value += ">"
        return value

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.uid != other.uid:
            return False
        if self.start != other.start:
            return False
        if self.end != other.end:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def _load_hdf5(cls, handle, comm=None, **kwargs):
        """Load the base class session"""
        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        session_name = None
        session_uid = None
        session_start = None
        session_end = None
        if handle is not None:
            session_name = str(handle.attrs["session_name"])
            session_uid = int(handle.attrs["session_uid"])
            session_start = handle.attrs["session_start"]
            if str(session_start) == "NONE":
                session_start = None
            else:
                session_start = datetime.datetime.fromtimestamp(
                    float(handle.attrs["session_start"]),
                    tz=timezone.utc,
                )
            session_end = handle.attrs["session_end"]
            if str(session_end) == "NONE":
                session_end = None
            else:
                session_end = datetime.datetime.fromtimestamp(
                    float(handle.attrs["session_end"]),
                    tz=timezone.utc,
                )

        if need_bcast:
            session_name = comm.bcast(session_name, root=0)
            session_uid = comm.bcast(session_uid, root=0)
            session_start = comm.bcast(session_start, root=0)
            session_end = comm.bcast(session_end, root=0)

        return cls(session_name, uid=session_uid, start=session_start, end=session_end)

    @classmethod
    def load_hdf5(cls, handle, comm=None, **kwargs):
        """Load the session from an HDF5 group.

        Args:
            handle (h5py.Group):  The group containing the session information.
            comm (MPI.Comm):  If loading from a file, optional communicator.

        Returns:
            None

        """
        log = Logger.get()

        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        session_class_name = None
        if handle is not None:
            session_class_name = str(handle.attrs["session_class"])

        if need_bcast:
            session_class_name = comm.bcast(session_class_name, root=0)

        session_class = import_from_name(session_class_name)
        return session_class._load_hdf5(handle, comm=comm, **kwargs)

    def _save_hdf5(self, handle, comm=None, **kwargs):
        """Save the base class session"""
        if handle is not None:
            handle.attrs["session_name"] = self.name
            handle.attrs["session_uid"] = self.uid
            if self.start is None:
                handle.attrs["session_start"] = "NONE"
            else:
                handle.attrs["session_start"] = self.start.timestamp()
            if self.end is None:
                handle.attrs["session_end"] = "NONE"
            else:
                handle.attrs["session_end"] = self.end.timestamp()

    def save_hdf5(self, handle, comm=None, **kwargs):
        """Save the session to an HDF5 group.

        Args:
            handle (h5py.Group):  The parent group for saving session properties.
            comm (MPI.Comm):  If saving to a file, optional communicator to broadcast
                across.

        Returns:
            None

        """
        if handle is not None:
            handle.attrs["session_class"] = object_fullname(self.__class__)
        self._save_hdf5(handle, comm=comm, **kwargs)


class Telescope(object):
    """Class representing telescope properties for one observation.

    Args:
        name (str):  The name of the telescope.
        uid (int):  The Unique ID of the telescope.  If not specified, constructed from
            a hash of the site name.
        focalplane (Focalplane):  The focalplane for this observation.
        site (Site):  The site of the telescope for this observation.

    """

    def __init__(self, name, uid=None, focalplane=None, site=None):
        self.name = name
        self.uid = uid
        if self.uid is None:
            self.uid = name_UID(name)
        if not isinstance(focalplane, Focalplane):
            raise RuntimeError("focalplane should be a Focalplane class instance")
        self.focalplane = focalplane
        if not isinstance(site, Site):
            raise RuntimeError("site should be a Site class instance")
        self.site = site

    def __repr__(self):
        value = "<Telescope '{}': uid = {}, site = {}, ".format(
            self.name,
            self.uid,
            self.site,
        )
        value += "focalplane = {}".format(self.focalplane.__repr__())
        value += ">"
        return value

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.uid != other.uid:
            return False
        if self.site != other.site:
            return False
        if self.focalplane != other.focalplane:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def _load_hdf5(cls, handle, comm=None, **kwargs):
        """Load the base class telescope."""
        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        telescope_name = None
        telescope_uid = None
        if handle is not None:
            telescope_name = str(handle.attrs["telescope_name"])
            telescope_uid = int(handle.attrs["telescope_uid"])

        if need_bcast:
            telescope_name = comm.bcast(telescope_name, root=0)
            telescope_uid = comm.bcast(telescope_uid, root=0)

        focalplane = Focalplane.load_hdf5(handle, comm=comm, **kwargs)
        site = Site.load_hdf5(handle, comm=comm, **kwargs)

        return cls(telescope_name, uid=telescope_uid, focalplane=focalplane, site=site)

    @classmethod
    def load_hdf5(cls, handle, comm=None, **kwargs):
        """Load the telescope from an HDF5 group.

        Args:
            handle (h5py.Group):  The group containing the telescope information.
            comm (MPI.Comm):  If loading from a file, optional communicator.

        Returns:
            None

        """
        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm) and comm is not None

        tele_class_name = None
        if handle is not None:
            tele_class_name = str(handle.attrs["telescope_class"])

        if need_bcast:
            tele_class_name = comm.bcast(tele_class_name, root=0)

        tele_class = import_from_name(tele_class_name)
        return tele_class._load_hdf5(handle, comm=comm, **kwargs)

    def _save_hdf5(self, handle, comm=None, **kwargs):
        """Load a base class telescope"""
        if handle is not None:
            handle.attrs["telescope_name"] = self.name
            handle.attrs["telescope_uid"] = self.uid
        self.focalplane.save_hdf5(handle, comm=comm, **kwargs)
        self.site.save_hdf5(handle, comm=comm, **kwargs)

    def save_hdf5(self, handle, comm=None, **kwargs):
        """Save the telescope to an HDF5 group.

        Args:
            handle (h5py.Group):  The parent group for saving telescope properties.
            comm (MPI.Comm):  If saving to a file, optional communicator.

        Returns:
            None

        """
        if handle is not None:
            handle.attrs["telescope_class"] = object_fullname(self.__class__)
        self._save_hdf5(handle, comm=comm, **kwargs)
