# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

import scipy.fftpack as sft
import scipy.interpolate as si
import scipy.sparse as sp

import healpy as hp

import quaternionarray as qa

from .tod import TOD

from .noise import Noise

from ..operator import Operator



class AnalyticNoise(Noise):
    """
    Class representing an analytic noise model.

    This generates an analytic PSD for a set of detectors, given
    input values for the knee frequency, NET, exponent, sample rate,
    minimum frequency, etc.

    Args:
        rate (float): sample rate in Hertz.
        fmin (float): minimum frequency for high pass
        detectors (list): list of detectors.
        fknee (dict): dictionary of knee frequencies.
        alpha (dict): dictionary of alpha exponents (positive, not negative!).
        NET (dict): dictionary of detector NETs.
    """

    def __init__(self, rate=None, fmin=None, detectors=None, fknee=None, alpha=None, NET=None):
        if rate is None:
            raise RuntimeError("you must specify the sample rate")
        if fmin is None:
            raise RuntimeError("you must specify the frequency for high pass")
        if detectors is None:
            raise RuntimeError("you must specify the detector list")
        if fknee is None:
            raise RuntimeError("you must specify the knee frequencies")
        if alpha is None:
            raise RuntimeError("you must specify the exponents")
        if NET is None:
            raise RuntimeError("you must specify the NET")

        self._rate = rate
        self._fmin = fmin
        self._detectors = detectors
        
        self._fknee = fknee
        self._alpha = alpha
        self._NET = NET

        for d in detectors:
            if self._alpha[d] < 0.0:
                raise RuntimeError("alpha exponents should be positive in this formalism")

        # for purposes of determining the common frequency sampling
        # points, use the lowest knee frequency.
        lowknee = rate
        for d in detectors:
            if self._fknee[d] < lowknee:
                lowknee = self._fknee[d]

        tempfreq = []
        cur = self._fmin
        while cur < 10.0 * lowknee:
            tempfreq.append(cur)
            cur *= 2.0
        nyquist = self._rate / 2.0
        df = (nyquist - cur) / 10.0
        tempfreq.extend([ (cur+x*df) for x in range(11) ])
        freq = np.array(tempfreq, dtype=np.float64)

        psds = {}

        for d in self._detectors:
            if self._fknee[d] > 0:
                ktemp = np.power(self._fknee[d], self._alpha[d])
                mtemp = np.power(self._fmin, self._alpha[d])
                temp = np.power(freq, self._alpha[d])
                psds[d] = (temp + ktemp) / (temp + mtemp)
                psds[d] *= (self._NET[d] * self._NET[d])
            else:
                psds[d] = np.ones_like(freq)
                psds[d] *= (self._NET[d] * self._NET[d])

        # call the parent class constructor to store the psds
        super().__init__(detectors=detectors, freq=freq, psds=psds)

    @property
    def fmin(self):
        return self._fmin

    def fknee(self, det):
        return self._fknee[det]

    def alpha(self, det):
        return self._alpha[det]

    def NET(self, det):
        return self._NET[det]

