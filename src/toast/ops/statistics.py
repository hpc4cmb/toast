# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re
from time import time
import warnings

from astropy import units as u
import h5py
import numpy as np
import traitlets

from ..mpi import MPI, MPI_Comm, use_mpi, Comm

from .operator import Operator
from .. import qarray as qa
from ..timing import function_timer
from ..traits import trait_docs, Int, Unicode, Bool, Dict, Quantity, Instance
from ..utils import Logger, Environment, Timer, GlobalTimers, dtype_to_aligned


@trait_docs
class Statistics(Operator):
    """Operator to measure and write out data statistics"""

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode("signal", help="Observation detdata key to analyze")

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(1, help="Bit mask value for optional detector flagging")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(1, help="Bit mask value for optional shared flagging")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    prefix = Unicode("stats", allow_none=True, help="Filename prefix to use")

    output_dir = Unicode(
        None,
        allow_none=True,
        help="If specified, write output data products to this directory",
    )

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        """Measure the statistics

        Args:
            data (toast.Data): The distributed data.

        """
        log = Logger.get()
        nstat = 5  # ngood, Mean, Variance, Skewness, Kurtosis

        for obs in data.obs:
            if obs.name is None:
                fname_out = f"{obs.uid}.h5"
            else:
                fname_out = f"{obs.name}.h5"
            if self.prefix is not None:
                fname_out = f"{self.prefix}_{fname_out}"
            if self.output_dir is not None:
                fname_out = os.path.join(self.output_dir, fname_out)
            all_dets = list(obs.all_detectors)
            ndet = len(all_dets)

            stats = np.zeros([nstat, ndet])

            obs_dets = obs.select_local_detectors(detectors)
            views = obs.view[self.view]
            for iview, view in enumerate(views):
                if self.shared_flags is not None:
                    shared_flags = views.shared[self.shared_flags][iview]
                    shared_mask = (shared_flags & self.shared_flag_mask) == 0
                else:
                    shared_mask = np.ones(nsample, dtype=bool)
                for det in obs_dets:
                    if self.det_flags is not None:
                        det_flags = views.detdata[self.det_flags][iview][det]
                        det_mask = (det_flags & self.det_flag_mask) == 0
                        mask = np.logical_and(shared_mask, det_mask)
                    else:
                        mask = shared_mask
                    ngood = np.sum(mask)
                    if ngood == 0:
                        continue
                    signal = views.detdata[self.det_data][iview][det]
                    good_signal = signal[mask].copy()
                    idet = all_dets.index(det)
                    # Valid samples
                    stats[0, idet] += ngood
                    # Mean
                    stats[1, idet] += np.sum(good_signal)
                    # Variance
                    stats[2, idet] += np.sum(good_signal ** 2)
                    # Skewness
                    stats[3, idet] += np.sum(good_signal ** 3)
                    # Kurtosis
                    stats[4, idet] += np.sum(good_signal ** 4)

            if obs.comm is not None:
                # Valid samples
                stats[0] = obs.comm.reduce(stats[0], op=MPI.SUM)
                # Mean
                stats[1] = obs.comm.reduce(stats[1], op=MPI.SUM)
                # Variance
                stats[2] = obs.comm.reduce(stats[2], op=MPI.SUM)
                # Skewness
                stats[3] = obs.comm.reduce(stats[3], op=MPI.SUM)
                # Kurtosis
                stats[4] = obs.comm.reduce(stats[4], op=MPI.SUM)
            if obs.comm.rank is None or obs.comm.rank == 0:
                good = stats[0] != 0
                hits = stats[0][good]
                # Mean
                stats[1][good] /= hits
                mean = stats[1][good]
                # Variance
                stats[2][good] = stats[2][good] / hits - mean ** 2
                var = stats[2][good]
                # Skewness
                stats[3][good] = stats[3][good] / hits - mean * var - mean ** 3
                skew = stats[3][good]
                # Kurtosis
                stats[4][good] = (
                    stats[4][good] / hits
                    - 4 * skew * mean
                    + 6 * var * mean ** 2
                    - 3 * mean ** 4
                )
                kurt = stats[4][good]
                # Normalize
                skew /= var ** 3 / 2
                kurt /= var ** 2
                # Write the results
                with h5py.File(fname_out, "w") as fout:
                    fout.attrs["UID"] = obs.uid
                    if obs.name is not None:
                        fout.attrs["name"] = obs.name
                    fout.attrs["nsample"] = obs.n_all_samples
                    fout["detectors"] = all_dets
                    fout["ngood"] = stats[0].astype(int)
                    fout["mean"] = stats[1]
                    fout["variance"] = stats[2]
                    fout["skewness"] = stats[3]
                    fout["kurtosis"] = stats[4]
                log.debug(f"Wrote data statistics to {fname_out}")

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov

    def _accelerators(self):
        return list()
