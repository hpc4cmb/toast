# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import pickle
import sys

import numpy as np

from .. import qarray
from ..timing import Timer, function_timer
from ..tod import AnalyticNoise
from ..utils import Environment, Logger

XAXIS, YAXIS, ZAXIS = np.eye(3)


def name2id(name, maxval=2**16):
    """Map a name into an index."""
    value = 0
    for c in name:
        value += ord(c)
    return value % maxval


class Focalplane:
    _detweights = None
    _detquats = None
    _noise = None

    def __init__(
        self, detector_data=None, fname_pickle=None, sample_rate=None, radius_deg=None
    ):
        """Instantiate a focalplane

        Args:
            detector_data (dict) :  Dictionary of detector attributes, such
                as detector quaternions and noise parameters.
            fname_pickle (str) :  Pickle file containing the focal
                 plane dictionary.  If both `detector_data` and
                 `fname_pickle` are set, the dictionaries are merged.
            sample_rate (float) :  Default sampling rate for all
                detectors.  Will be overridden by 'fsample' fields
                if they exist for the detectors in the dictionary.
            radius_deg (float) :  force the radius of the focal plane.
                otherwise it will be calculated from the detector
                offsets.
        """
        self.detector_data = {}
        if detector_data is not None:
            self.detector_data.update(detector_data)
        if fname_pickle is not None:
            with open(fname_pickle, "rb") as picklefile:
                self.detector_data.update(pickle.load(picklefile))
        self.sample_rate = sample_rate
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
        """Get the polarization efficiency from polarization leakage
        or vice versa
        """
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

    def reset_properties(self):
        """Clear automatic properties so they will be re-generated"""
        self._detweights = None
        self._radius = None
        self._detquats = None
        self._noise = None

    @property
    def detweights(self):
        """Return the inverse noise variance weights [K_CMB^-2]"""
        if self._detweights is None:
            self._detweights = {}
            for detname, detdata in self.detector_data.items():
                net = detdata["NET"]
                if "fsample" in detdata:
                    fsample = detdata["fsample"]
                else:
                    fsample = self.sample_rate
                detweight = 1.0 / (fsample * net**2)
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
    def detquats(self):
        if self._detquats is None:
            self._detquats = {}
            for detname, detdata in self.detector_data.items():
                self._detquats[detname] = detdata["quat"]
        return self._detquats

    @property
    def noise(self):
        if self._noise is None:
            detectors = sorted(self.detector_data.keys())
            fmin = {}
            fknee = {}
            alpha = {}
            NET = {}
            rates = {}
            for detname in detectors:
                detdata = self.detector_data[detname]
                if "fsample" in detdata:
                    rates[detname] = detdata["fsample"]
                else:
                    rates[detname] = self.sample_rate
                fmin[detname] = detdata["fmin"]
                fknee[detname] = detdata["fknee"]
                alpha[detname] = detdata["alpha"]
                NET[detname] = detdata["NET"]
            self._noise = AnalyticNoise(
                rate=rates,
                fmin=fmin,
                detectors=detectors,
                fknee=fknee,
                alpha=alpha,
                NET=NET,
            )
        return self._noise

    def __repr__(self):
        value = (
            "(Focalplane : {} detectors, sample_rate = {} Hz, radius = {} deg, "
            "detectors = ("
            "".format(len(self.detector_data), self.sample_rate, self.radius)
        )
        for detector_name, detector_data in self.detector_data.items():
            value += "{}, ".format(detector_name)
        value += "))"
        return value


class Telescope(object):
    def __init__(self, name, focalplane=None, site=None):
        self.name = name
        self.id = name2id(name)
        self.focalplane = focalplane
        self.site = site

    def __repr__(self):
        value = "(Telescope '{}' : ID = {}, Site = {}, Focalplane = {}" "".format(
            self.name, self.id, self.site, self.focalplane
        )
        return value
