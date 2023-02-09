# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import h5py
import numpy as np
from astropy import units as u

from .noise import Noise
from .timing import function_timer
from .utils import Logger, hdf5_use_serial


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
        # Convert everything to consistent units
        self._rate = {x: y.to(u.Hz) for x, y in rate.items()}
        self._fmin = {x: y.to(u.Hz) for x, y in fmin.items()}
        self._fknee = {x: y.to(u.Hz) for x, y in fknee.items()}
        self._alpha = dict(alpha)
        self._NET = {x: y.to(u.K * np.sqrt(1.0 * u.second)) for x, y in NET.items()}

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
                psd_vals = (temp + ktemp) / (temp + mtemp)
                psds[d] = psd_vals * self._NET[d] ** 2
            else:
                psd_vals = np.ones_like(freqs[d].to_value(u.Hz))
                psds[d] = psd_vals * self._NET[d] ** 2

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
        wt = 1.0 / (self._NET[det] ** 2) / self._rate[det]
        return wt.decompose()

    def _gather_sim(self, comm, out):
        # Gather simulation properties to the rank zero process.
        # The output data dictionary is modified in place.
        if comm is None or comm.size == 1:
            out["fmin"] = self._fmin
            out["fknee"] = self._fknee
            out["rate"] = self._rate
            out["alpha"] = self._alpha
            out["NET"] = self._NET
            return

        pfmin = comm.gather(self._fmin, root=0)
        pfknee = comm.gather(self._fknee, root=0)
        prate = comm.gather(self._rate, root=0)
        palpha = comm.gather(self._alpha, root=0)
        pnet = comm.gather(self._NET, root=0)
        if comm.rank == 0:
            out["fmin"] = dict()
            out["fknee"] = dict()
            out["rate"] = dict()
            out["alpha"] = dict()
            out["NET"] = dict()
            for pr, pfm, pfk, pa, pn in zip(prate, pfmin, pfknee, palpha, pnet):
                out["rate"].update(pr)
                out["fmin"].update(pfm)
                out["fknee"].update(pfk)
                out["alpha"].update(pa)
                out["NET"].update(pn)

    def _gather(self, comm):
        # Base class properties
        out = self._gather_base(comm)
        # Sim properties
        self._gather_sim(comm, out)
        return out

    def _scatter_sim(self, comm, local_dets, props):
        if comm is None or comm.size == 1:
            self._fmin = props["fmin"]
            self._fknee = props["fknee"]
            self._rate = props["rate"]
            self._alpha = props["alpha"]
            self._NET = props["NET"]
            return

        # Broadcast and extract our local properties
        all_rate = None
        all_fmin = None
        all_fknee = None
        all_alpha = None
        all_net = None
        if comm.rank == 0:
            all_rate = props["rate"]
            all_fmin = props["fmin"]
            all_fknee = props["fknee"]
            all_alpha = props["alpha"]
            all_net = props["NET"]

        all_rate = comm.bcast(all_rate, root=0)
        self._rate = {x: all_rate[x] for x in local_dets}
        del all_rate

        all_fmin = comm.bcast(all_fmin, root=0)
        self._fmin = {x: all_fmin[x] for x in local_dets}
        del all_fmin

        all_fknee = comm.bcast(all_fknee, root=0)
        self._fknee = {x: all_fknee[x] for x in local_dets}
        del all_fknee

        all_alpha = comm.bcast(all_alpha, root=0)
        self._alpha = {x: all_alpha[x] for x in local_dets}
        del all_alpha

        all_net = comm.bcast(all_net, root=0)
        self._NET = {x: all_net[x] for x in local_dets}
        del all_net

    def _scatter(self, comm, local_dets, props):
        self._scatter_base(comm, local_dets, props)
        self._scatter_sim(comm, local_dets, props)

    def _save_hdf5(self, handle, obs, **kwargs):
        gcomm = obs.comm.comm_group
        rank = 0
        if gcomm is not None:
            rank = gcomm.rank

        # First save the base class info
        self._save_base_hdf5(handle, obs)

        # Each column of the process grid has the same information.
        props = None
        maxstr = 0
        n_det = 0
        if obs.comm_row_rank == 0:
            props = dict()
            self._gather_sim(obs.comm_col, props)
            if obs.comm_col_rank == 0:
                n_det = len(props["NET"])
                maxstr = 1 + max([len(x) for x in props["NET"].keys()])

        if gcomm is not None:
            n_det = gcomm.bcast(n_det, root=0)
            maxstr = gcomm.bcast(maxstr, root=0)

        ds = None
        if handle is not None:
            # Write the noise model parameters as a dataset
            adtype = np.dtype(f"a{maxstr}, f8, f8, f8, f8, f8")
            ds = handle.create_dataset("analytic", (n_det,), dtype=adtype)
        if gcomm is not None:
            gcomm.barrier()

        if rank == 0:
            packed = np.array(
                [
                    (
                        d,
                        props["rate"][d].to_value(u.Hz),
                        props["fmin"][d].to_value(u.Hz),
                        props["fknee"][d].to_value(u.Hz),
                        props["alpha"][d],
                        props["NET"][d].to_value(u.K * np.sqrt(1.0 * u.second)),
                    )
                    for d in props["NET"].keys()
                ],
                dtype=adtype,
            )
            ds.write_direct(packed)
        if gcomm is not None:
            gcomm.barrier()

        del ds

    def _load_hdf5(self, handle, obs, **kwargs):
        # First load the base class information
        self._load_base_hdf5(handle, obs)

        gcomm = obs.comm.comm_group
        rank = 0
        if gcomm is not None:
            rank = gcomm.rank

        props = None
        if handle is not None:
            ds = handle["analytic"]
            if rank == 0:
                # get noise model parameters
                props = dict()
                props["rate"] = dict()
                props["fmin"] = dict()
                props["fknee"] = dict()
                props["alpha"] = dict()
                props["NET"] = dict()
                for row in ds[:]:
                    dname = row[0].decode()
                    props["rate"][dname] = row[1] * u.Hz
                    props["fmin"][dname] = row[2] * u.Hz
                    props["fknee"][dname] = row[3] * u.Hz
                    props["alpha"][dname] = row[4]
                    props["NET"][dname] = row[5] * u.K * np.sqrt(1.0 * u.second)
            del ds

        # The data now exists on the rank zero process of the group.  If we have
        # multiple processes along each row, broadcast data to the other processes
        # in the first row.
        if obs.comm_row_size > 1 and obs.comm_col_rank == 0:
            props = obs.comm_row.bcast(props, root=0)

        # Scatter across each column of the process grid
        self._scatter_sim(
            obs.comm_col,
            obs.local_detectors,
            props,
        )

    def __repr__(self):
        value = f"<AnalyticNoise model with {len(self._dets)} detectors"
        value += ">"
        return value

    def __eq__(self, other):
        log = Logger.get()
        fail = 0
        if not super().__eq__(other):
            # Base class values not equal
            fail = 1
        elif self._rate != other._rate:
            log.verbose(f"AnalyticNoise __eq__:  rate {self._rate} != {other._rate}")
            fail = 1
        elif self._fmin != other._fmin:
            log.verbose(f"AnalyticNoise __eq__:  fmin {self._fmin} != {other._fmin}")
            fail = 1
        elif self._fknee != other._fknee:
            log.verbose(f"AnalyticNoise __eq__:  fknee {self._fknee} != {other._fknee}")
            fail = 1
        elif self._alpha != other._alpha:
            log.verbose(f"AnalyticNoise __eq__:  alpha {self._alpha} != {other._alpha}")
            fail = 1
        elif self._NET != other._NET:
            log.verbose(f"AnalyticNoise __eq__:  NET {self._NET} != {other._NET}")
            fail = 1
        return fail == 0

    def __ne__(self, other):
        return not self.__eq__(other)
