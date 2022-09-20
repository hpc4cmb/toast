# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import h5py
import numpy as np
from astropy import units as u

from .noise import Noise
from .timing import function_timer
from .utils import hdf5_use_serial


class AnalyticNoise(Noise):
    """Class representing an analytic noise model.

    This generates an analytic PSD for a set of detectors, given input values for the
    knee frequency, NET, exponent, sample rate, minimum frequency, etc.  The rate,
    fmin, fknee, and NET parameters should all be Quantities with units.

    Args:
        detectors (list): List of detectors.
        rate (dict): Dictionary of sample rates.
        fmin (dict): Dictionary of minimum frequencies for high pass
        fknee (dict): Dictionary of knee frequencies.
        alpha (dict): Dictionary of alpha exponents (positive, not negative!).
        NET (dict): Dictionary of detector NETs.

    """

    @function_timer
    def __init__(
        self,
        detectors=list(),
        rate=dict(),
        fmin=dict(),
        fknee=dict(),
        alpha=dict(),
        NET=dict(),
        indices=None,
    ):
        self._rate = rate
        self._fmin = fmin
        self._fknee = fknee
        self._alpha = alpha
        self._NET = NET

        for d in detectors:
            if self._alpha[d] < 0.0:
                raise RuntimeError(
                    "alpha exponents should be positive in this formalism"
                )

        freqs = {}
        psds = {}

        last_nyquist = None
        last_det = None

        for d in detectors:
            if last_det is not None:
                # shortcut when the noise models are identical
                if (
                    self._rate[d] == self._rate[last_det]
                    and self._fmin[d] == self._fmin[last_det]
                    and self._fknee[d] == self._fknee[last_det]
                    and self._alpha[d] == self._alpha[last_det]
                    and self._NET[d] == self._NET[last_det]
                ):
                    freqs[d] = freqs[last_det].copy()
                    psds[d] = psds[last_det].copy()
                    continue

            fmin_hz = self._fmin[d].to_value(u.Hz)
            fknee_hz = self._fknee[d].to_value(u.Hz)
            rate_hz = self._rate[d].to_value(u.Hz)
            if (fknee_hz > 0.0) and (fknee_hz < fmin_hz):
                raise RuntimeError(
                    "If knee frequency is non-zero, it must be greater than f_min"
                )

            nyquist = rate_hz / 2.0
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

            freqs[d] = tempfreq * u.Hz

            if fknee_hz > 0.0:
                ktemp = np.power(fknee_hz, self._alpha[d])
                mtemp = np.power(fmin_hz, self._alpha[d])
                temp = np.power(freqs[d].to_value(u.Hz), self._alpha[d])
                psds[d] = (temp + ktemp) / (temp + mtemp)
                psds[d] *= (self._NET[d].to_value(u.K * np.sqrt(1.0 * u.second))) ** 2
            else:
                psds[d] = np.ones_like(freqs[d].to_value(u.Hz))
                psds[d] *= (self._NET[d].to_value(u.K * np.sqrt(1.0 * u.second))) ** 2
            psds[d] *= (self._NET[d].unit) ** 2

            last_det = d

        # call the parent class constructor to store the psds
        super().__init__(detectors=detectors, freqs=freqs, psds=psds, indices=indices)

    def fmin(self, det):
        """(float): the minimum frequency, used as a high pass."""
        return self._fmin[det]

    def fknee(self, det):
        """(float): the knee frequency."""
        return self._fknee[det]

    def alpha(self, det):
        """(float): the (positive!) slope exponent."""
        return self._alpha[det]

    def NET(self, det):
        """(float): the NET."""
        return self._NET[det]

    def _detector_weight(self, det):
        return (
            1.0
            / (self._NET[det] ** 2).to_value(u.K**2 * u.second)
            / self._rate[det].to_value(u.Hz)
        )

    def _save_hdf5(self, handle, comm, **kwargs):
        """Internal method which can be overridden by derived classes."""
        # First save the base class info
        self._save_base_hdf5(handle, comm)

        rank = 0
        if comm is not None:
            rank = comm.rank

        if handle is not None:
            # Write the noise model parameters as a dataset
            maxstr = 1 + max([len(x) for x in self._dets])
            adtype = np.dtype(f"a{maxstr}, f8, f8, f8, f8, f8")
            ds = handle.create_dataset("analytic", (len(self._dets),), dtype=adtype)
            if rank == 0:
                packed = np.array(
                    [
                        (
                            d,
                            self._rate[d].to_value(u.Hz),
                            self._fmin[d].to_value(u.Hz),
                            self._fknee[d].to_value(u.Hz),
                            self._alpha[d],
                            self._NET[d].to_value(u.K * np.sqrt(1.0 * u.second)),
                        )
                        for d in self._dets
                    ],
                    dtype=adtype,
                )
                ds.write_direct(packed)
            del ds

    def _load_hdf5(self, handle, comm, **kwargs):
        """Internal method which can be overridden by derived classes."""
        # First load the base class information
        self._load_base_hdf5(handle, comm)

        # Now load the dataset of analytic parameters.

        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(handle, comm)

        if handle is not None:
            # get noise model paramters
            self._rate = dict()
            self._fmin = dict()
            self._fknee = dict()
            self._alpha = dict()
            self._NET = dict()
            ds = handle["analytic"]
            for row in ds[:]:
                dname = row[0].decode()
                self._rate[dname] = row[1] * u.Hz
                self._fmin[dname] = row[2] * u.Hz
                self._fknee[dname] = row[3] * u.Hz
                self._alpha[dname] = row[4]
                self._NET[dname] = row[5] * u.K * np.sqrt(1.0 * u.second)
            del ds

        if need_bcast and comm is not None:
            self._rate = comm.bcast(self._rate, root=0)
            self._fmin = comm.bcast(self._fmin, root=0)
            self._fknee = comm.bcast(self._fknee, root=0)
            self._alpha = comm.bcast(self._alpha, root=0)
            self._NET = comm.bcast(self._NET, root=0)

    def __repr__(self):
        value = f"<AnalyticNoise model with {len(self._dets)} detectors"
        value += ">"
        return value

    def __eq__(self, other):
        if not super().__eq__(other):
            # Base class values not equal
            return False
        if self._rate != other._rate:
            return False
        if self._fmin != other._fmin:
            return False
        if self._fknee != other._fknee:
            return False
        if self._alpha != other._alpha:
            return False
        if self._NET != other._NET:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)
