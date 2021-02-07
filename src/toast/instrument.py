# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import numpy as np

from astropy import units as u

import tomlkit

from .timing import function_timer, Timer

from .utils import Logger, Environment, name_UID

from . import qarray

# FIXME: This Focalplane class should be much more generic and able to hold common
# metadata as well as per-detector properties.  We should inherit from MutableMapping
# to support full dictionary behavior and then have a special key for the
# detector-specific properties.  Unfortunately, this will break the API, so should
# be done as part of the 3.0 transition.

XAXIS, YAXIS, ZAXIS = np.eye(3)


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

    def __repr__(self):
        value = "<Site '{}' : uid = {}>".format(self.name, self.uid)
        return value


class GroundSite(Site):
    """Site on the Earth.

    This represents a fixed location on the Earth.

    Args:
        name (str):  Site name
        lat (str):  Site latitude as a pyEphem string
        lon (str):  Site longitude as a pyEphem string
        alt (Quantity):  Site altitude.
        uid (int):  Unique identifier.  If not specified, constructed from a hash
            of the site name.
        weather (Weather):  Weather information for this site.
    """

    def __init__(self, name, lat, lon, alt, uid=None, weather=None):
        super().__init__(name, uid)
        # Strings get interpreted correctly by pyEphem.
        # Floats must be in radians
        self.lat = str(lat)
        self.lon = str(lon)
        self.alt = alt.to_value(u.meter)
        self.weather = weather

    def __repr__(self):
        value = "<GroundSite '{}' : uid = {}, lon = {}, lat = {}, alt = {} m, weather = {}>".format(
            self.name, self.uid, self.lon, self.lat, self.alt, self.weather
        )
        return value


class SpaceSite(Site):
    """Site with no atmosphere.

    This represents a location beyond the Earth's atmosphere.  In practice, this
    should be sub-classed for real satellite experiments.  However, we keep this here
    in case we wish to implement generic satellite location properties.

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


class Focalplane(object):
    """Class representing the focalplane for one observation.

    Args:
        detector_data (dict):  Dictionary of detector attributes, such
            as detector quaternions and noise parameters.
        radius_deg (float):  force the radius of the focal plane.
            otherwise it will be calculated from the detector
            offsets.
        sample_rate (float):  The common (nominal) sample rate for all detectors.
        fname (str):  Load the focalplane from this file.

    """

    def __init__(
        self, detector_data=None, radius_deg=None, sample_rate=None, fname=None
    ):
        self.detector_data = None
        self.sample_rate = None
        self._radius = None
        self._detweights = None
        self._detquats = None
        self._noise = None

        if fname is not None:
            raw = None
            with open(fname, "r") as f:
                raw = tomlkit.loads(f.read())
            self.sample_rate = raw["sample_rate"]
            if "radius" in raw:
                self._radius = raw["radius"]
            self.detector_data = raw["detector_data"]
        else:
            if detector_data is None:
                raise RuntimeError(
                    "If not loading from a file, must specify detector_data"
                )
            self.detector_data = detector_data
            if sample_rate is None:
                raise RuntimeError(
                    "If not loading from a file, must specify sample_rate"
                )
            self.sample_rate = sample_rate
            if radius_deg is not None:
                self._radius = radius_deg

        self._get_pol_angles()
        self._get_pol_efficiency()

    def _get_pol_angles(self):
        """Get the detector polarization angles from the quaternions"""
        for detname, detdata in self.detector_data.items():
            if "pol_angle_deg" in detdata or "pol_angle_rad" in detdata:
                continue
            quat = detdata["quat"]
            theta, phi = qarray.to_position(quat)
            yrot = qarray.rotation(YAXIS, -theta)
            zrot = qarray.rotation(ZAXIS, -phi)
            rot = qarray.norm(qarray.mult(yrot, zrot))
            pol_rot = qarray.mult(rot, quat)
            pol_angle = qarray.to_angles(pol_rot)[2]
            detdata["pol_angle_rad"] = pol_angle
        return

    def _get_pol_efficiency(self):
        """Get the polarization efficiency from polarization leakage or vice versa"""
        for detname, detdata in self.detector_data.items():
            if "pol_leakage" in detdata and "pol_efficiency" not in detdata:
                # Derive efficiency from leakage
                epsilon = detdata["pol_leakage"]
                eta = (1 - epsilon) / (1 + epsilon)
                detdata["pol_efficiency"] = eta
            elif "pol_leakage" not in detdata and "pol_efficiency" in detdata:
                # Derive leakage from efficiency
                eta = detdata["pol_effiency"]
                epsilon = (1 - eta) / (1 + eta)
                detdata["pol_leakage"] = epsilon
            elif "pol_leakage" not in detdata and "pol_efficiency" not in detdata:
                # Assume a perfectly polarized detector
                detdata["pol_efficiency"] = 1
                detdata["pol_leakage"] = 0
            else:
                # Check that efficiency and leakage are consistent
                epsilon = detdata["pol_leakage"]
                eta = detdata["pol_efficiency"]
                np.testing.assert_almost_equal(
                    eta,
                    (1 + epsilon) / (1 - epsilon),
                    err_msg="inconsistent polarization leakage and efficiency",
                )
        return

    def __contains__(self, key):
        return key in self.detector_data

    def __getitem__(self, key):
        return self.detector_data[key]

    def __setitem__(self, key, value):
        self.detector_data[key] = value
        if "UID" not in value:
            self.detector_data[key]["UID"] = name_UID(key)

    def reset_properties(self):
        """Clear automatic properties so they will be re-generated"""
        self._detweights = None
        self._radius = None
        self._detquats = None
        self._noise = None

    @property
    def detectors(self):
        return sorted(self.detector_data.keys())

    def keys(self):
        return self.detectors

    @property
    def detector_index(self):
        return {name: props["UID"] for name, props in self.detector_data.items()}

    @property
    def detector_weights(self):
        """Return the inverse noise variance weights [K_CMB^-2]"""
        if self._detweights is None:
            self._detweights = {}
            for detname, detdata in self.detector_data.items():
                net = detdata["NET"]
                if "fsample" in detdata:
                    fsample = detdata["fsample"]
                else:
                    fsample = self.sample_rate
                detweight = 1.0 / (fsample * net ** 2)
                self._detweights[detname] = detweight
        return self._detweights

    @property
    def radius(self):
        """The focal plane radius in degrees"""
        if self._radius is None:
            # Find the largest distance from the bore sight
            ZAXIS = np.array([0, 0, 1])
            cosangs = []
            for detname, detdata in self.detector_data.items():
                quat = detdata["quat"]
                vec = qarray.rotate(quat, ZAXIS)
                cosangs.append(np.dot(ZAXIS, vec))
            mincos = np.amin(cosangs)
            self._radius = np.degrees(np.arccos(mincos))
            # Add a very small margin to avoid numeric issues
            # in the atmospheric simulation
            self._radius *= 1.001
        return self._radius

    @property
    def detector_quats(self):
        if self._detquats is None:
            self._detquats = {}
            for detname, detdata in self.detector_data.items():
                self._detquats[detname] = detdata["quat"]
        return self._detquats

    def __repr__(self):
        value = "<Focalplane: {} detectors, sample_rate = {} Hz, radius = {} deg, detectors = [".format(
            len(self.detector_data), self.sample_rate, self.radius
        )
        value += "{} .. {}".format(self.detectors[0], self.detectors[-1])
        value += "]>"
        return value


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
