# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re
import warnings
from time import time

import h5py
import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..mpi import MPI, Comm, MPI_Comm, use_mpi
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Int, Unicode, trait_docs
from ..utils import Environment, GlobalTimers, Logger, Timer, dtype_to_aligned
from .operator import Operator


@trait_docs
class Statistics(Operator):
    """Operator to measure and write out data statistics"""

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(defaults.det_data, help="Observation detdata key to analyze")

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

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
        nstat = 3  # Variance, Skewness, Kurtosis

        for obs in data.obs:
            # NOTE:  We could use the session name / uid in the filename
            # too for easy sorting.
            if obs.name is None:
                fname_out = f"{self.name}_{obs.uid}.h5"
            else:
                fname_out = f"{self.name}_{obs.name}.h5"
            if self.output_dir is not None:
                fname_out = os.path.join(self.output_dir, fname_out)
            all_dets = list(obs.all_detectors)
            ndet = len(all_dets)

            hits = np.zeros([ndet], dtype=int)
            means = np.zeros([ndet], dtype=float)
            stats = np.zeros([nstat, ndet], dtype=float)

            obs_dets = obs.select_local_detectors(detectors)
            views = obs.view[self.view]

            # Measure the mean separately to simplify the math
            for iview, view in enumerate(views):
                if view.start is None:
                    # This is a view of the whole obs
                    nsample = obs.n_local_samples
                else:
                    nsample = view.stop - view.start
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
                    hits[idet] += ngood
                    # Mean
                    means[idet] += np.sum(good_signal)

            if obs.comm.comm_group is not None:
                hits = obs.comm.comm_group.allreduce(hits, op=MPI.SUM)
                means = obs.comm.comm_group.allreduce(means, op=MPI.SUM)

            good = hits != 0
            means[good] /= hits[good]

            # Now evaluate the moments

            for iview, view in enumerate(views):
                if view.start is None:
                    # This is a view of the whole obs
                    nsample = obs.n_local_samples
                else:
                    nsample = view.stop - view.start
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
                    idet = all_dets.index(det)
                    signal = views.detdata[self.det_data][iview][det]
                    good_signal = signal[mask].copy() - means[idet]
                    # Variance
                    stats[0, idet] += np.sum(good_signal**2)
                    # Skewness
                    stats[1, idet] += np.sum(good_signal**3)
                    # Kurtosis
                    stats[2, idet] += np.sum(good_signal**4)

            if obs.comm.comm_group is not None:
                stats = obs.comm.comm_group.reduce(stats, op=MPI.SUM)

            if obs.comm.group_rank == 0:
                # Central moments
                m2 = stats[0]
                m3 = stats[1]
                m4 = stats[2]
                for m in m2, m3, m4:
                    m[good] /= hits[good]
                # Variance
                var = m2.copy()
                # Skewness
                skew = m3.copy()
                skew[good] /= m2[good] ** 1.5
                # Kurtosis
                kurt = m4.copy()
                kurt[good] /= m2[good] ** 2
                # Write the results
                with h5py.File(fname_out, "w") as fout:
                    fout.attrs["UID"] = obs.uid
                    if obs.name is not None:
                        fout.attrs["name"] = obs.name
                    fout.attrs["nsample"] = obs.n_all_samples
                    fout.create_dataset(
                        "detectors", data=all_dets, dtype=h5py.string_dtype()
                    )
                    fout["ngood"] = hits
                    fout["mean"] = means
                    fout["variance"] = var
                    fout["skewness"] = skew
                    fout["kurtosis"] = kurt
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
