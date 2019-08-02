# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import pickle
import sys

import numpy as np

from ..timing import function_timer, Timer
from ..tod import AnalyticNoise
from ..utils import Logger, Environment
from .. import qarray


def name2id(name, maxval=2 ** 16):
    """ Map a name into an index.
    """
    value = 0
    for c in name:
        value += ord(c)
    return value % maxval


class Focalplane:
    detector_data = None
    _detweights = None
    sample_rate = None
    _radius = None
    _detquats = None
    _noise = None

    def __init__(
        self, detector_data=None, fname_pickle=None, sample_rate=None, radius_deg=None
    ):
        """ Instantiate a focalplane

        Args:
            fname_pickle (str) :  Pickle file containing the focal
                 plane dictionary
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
        return

    def refresh(self):
        """ Clear automatic properties so they will be re-generated
        """
        self._detweights = None
        self._radius = None
        self._detquats = None
        self._noise = None
        return

    @property
    def detweights(self):
        """ Return the inverse noise variance weights [K_CMB^-2]
        """
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
        """ The focal plane radius in degrees
        """
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
        value = "(Focalplane : {} detectors, sample_rate = {} Hz, radius = {} deg, " \
            "detectors = (" \
            "".format(len(self.detector_data), self.sample_rate, self.radius)
        for detector_name, detector_data in self.detector_data.items():
            value += "{}, ".format(detector_name)
        value += "))"
        return value


class Telescope(object):
    focalplane = None
    site = None

    def __init__(self, name, focalplane=None, site=None):
        self.name = name
        self.id = name2id(name)
        self.focalplane = focalplane
        self.site = site
        return

    def __repr__(self):
        value = "(Telescope '{}' : ID = {}, Site = {}, Focalplane = {}" \
            "".format(self.name, self.id, self.site, self.focalplane)
        return value
