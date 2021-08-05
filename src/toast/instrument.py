# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import numpy as np

from astropy import units as u

import astropy.time as astime

import astropy.coordinates as coord

from astropy.table import QTable, Column

import tomlkit

from .timing import function_timer, Timer

from . import qarray as qa

from .utils import Logger, Environment, name_UID

from . import qarray


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


class Focalplane(object):
    """Class representing the focalplane for one observation.

    The detector_data Table may store arbitrary columns, but several are required.
    They include:

        "name":  The detector name.
        "quat":  Each row should be a 4-element numpy array.

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

    Args:
        detector_data (QTable):  Table of detector properties.
        field_of_view (Quantity):  Angular diameter of the focal plane.  Used to
            increase the effective size of the focalplane when simulating atmosphere,
            etc.  Will be calculated from the detector offsets by default.
        sample_rate (Quantity):  The common (nominal) sample rate for all detectors.
        file (str):  Load the focalplane from this file.
        comm (MPI.Comm):  If loading from a file, optional communicator to broadcast
            across.

    """

    XAXIS, YAXIS, ZAXIS = np.eye(3)

    def __init__(
        self,
        detector_data=None,
        field_of_view=None,
        sample_rate=None,
        file=None,
        comm=None,
    ):
        self.detector_data = None
        self.field_of_view = None
        self.sample_rate = None

        if file is not None:
            self.read(file, comm=comm)
        else:
            self.detector_data = detector_data
            self.field_of_view = field_of_view
            self.sample_rate = sample_rate

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
                pol_angle = np.arctan2(2 * a * d, a ** 2 - d ** 2) % np.pi
                row["pol_angle"] = pol_angle * u.radian

    def _get_pol_efficiency(self):
        """Get the polarization efficiency from polarization leakage or vice versa"""

        n_rows = len(self.detector_data)
        if ("pol_leakage" in self.detector_data.colnames) and (
            "pol_efficiency" in self.detector_data.colnames
        ):
            # Check that efficiency and leakage are consistent
            epsilon = self.detector_data["pol_leakage"]
            eta = self.detector_data["pol_efficiency"]
            np.testing.assert_almost_equal(
                eta,
                (1 + epsilon) / (1 - epsilon),
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

    def keys(self):
        return self.detectors

    def __repr__(self):
        value = "<Focalplane: {} detectors, sample_rate = {} Hz, FOV = {} deg, detectors = [".format(
            len(self.detector_data),
            self.sample_rate.to_value(u.Hz),
            self.field_of_view.to_value(u.degree),
        )
        value += "{} .. {}".format(self.detectors[0], self.detectors[-1])
        value += "]>"
        return value

    def read(self, file, comm=None):
        if comm is None or comm.rank == 0:
            self.detector_data = QTable.read(file, format="hdf5", path="focalplane")
            self.sample_rate = self.detector_data.meta["sample_rate"]
            if "field_of_view" in self.detector_data.meta:
                self.field_of_view = self.detector_data.meta["field_of_view"]
            else:
                self._compute_fov()
        if comm is not None:
            self.detector_data = comm.bcast(self.detector_data, root=0)
            self.sample_rate = comm.bcast(self.sample_rate, root=0)
            self.field_of_view = comm.bcast(self.field_of_view, root=0)

    def write(self, file, comm=None):
        if comm is None or comm.rank == 0:
            self.detector_data.meta["sample_rate"] = self.sample_rate
            self.detector_data.meta["field_of_view"] = self.field_of_view
            self.detector_data.write(
                file,
                format="hdf5",
                path="focalplane",
                serialize_meta=True,
                overwrite=True,
            )


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
        value = "<Telescope '{}': uid = {}, site = {}, focalplane = ".format(
            self.name, self.uid, self.site
        )
        value += self.focalplane.__repr__()
        value += ">"
        return value
