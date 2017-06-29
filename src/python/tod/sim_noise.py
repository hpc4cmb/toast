# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

from .tod import TOD

from .noise import Noise

from ..op import Operator


class AnalyticNoise(Noise):
    """
    Class representing an analytic noise model.

    This generates an analytic PSD for a set of detectors, given
    input values for the knee frequency, NET, exponent, sample rate,
    minimum frequency, etc.

    Args:
        detectors (list): list of detectors.
        rate (dict): dictionary of sample rates in Hertz.
        fmin (dict): dictionary of minimum frequencies for high pass
        fknee (dict): dictionary of knee frequencies.
        alpha (dict): dictionary of alpha exponents (positive, not negative!).
        NET (dict): dictionary of detector NETs.
    """

    def __init__(self, detectors=None, rate=None, fmin=None, fknee=None, alpha=None, NET=None):
        if detectors is None:
            raise RuntimeError("you must specify the detector list")
        if rate is None:
            raise RuntimeError("you must specify the sample rates")
        if fmin is None:
            raise RuntimeError("you must specify the frequencies for high pass")
        if fknee is None:
            raise RuntimeError("you must specify the knee frequencies")
        if alpha is None:
            raise RuntimeError("you must specify the exponents")
        if NET is None:
            raise RuntimeError("you must specify the NET")

        self._detectors = detectors
        self._rate = rate
        self._fmin = fmin
        self._fknee = fknee
        self._alpha = alpha
        self._NET = NET

        for d in detectors:
            if self._alpha[d] < 0.0:
                raise RuntimeError("alpha exponents should be positive in this formalism")

        freqs = {}
        psds = {}

        last_nyquist = None

        for d in self._detectors:
            if (self._fknee[d] > 0.0) and (self._fknee[d] < self._fmin[d]):
                raise RuntimeError("If knee frequency is non-zero, it must be greater than f_min")

            nyquist = self._rate[d] / 2.0
            if nyquist != last_nyquist:
                tempfreq = []

                # this starting point corresponds to a high-pass of
                # 30 years, so should be low enough for any interpolation!
                cur = 1.0e-9

                # this value seems to provide a good density of points
                # in log space.
                while cur < nyquist:
                    tempfreq.append(cur)
                    cur *= 1.4

                # put a final point at Nyquist
                tempfreq.append(nyquist)
                tempfreq = np.array(tempfreq, dtype=np.float64)
                last_nyquist = nyquist

            freqs[d] = tempfreq

            if self._fknee[d] > 0.0:
                ktemp = np.power(self._fknee[d], self._alpha[d])
                mtemp = np.power(self._fmin[d], self._alpha[d])
                temp = np.power(freqs[d], self._alpha[d])
                psds[d] = (temp + ktemp) / (temp + mtemp)
                psds[d] *= (self._NET[d] * self._NET[d])
            else:
                psds[d] = np.ones_like(freqs[d])
                psds[d] *= (self._NET[d] * self._NET[d])

        # call the parent class constructor to store the psds
        super().__init__(detectors=detectors, freqs=freqs, psds=psds)

    @property
    def fmin(self, det):
        """
        (float): the minimum frequency in Hz, used as a high pass.
        """
        return self._fmin

    def fknee(self, det):
        """
        (float): the knee frequency in Hz.
        """
        return self._fknee[det]

    def alpha(self, det):
        """
        (float): the (positive!) slope exponent.
        """
        return self._alpha[det]

    def NET(self, det):
        """
        (float): the NET.
        """
        return self._NET[det]
