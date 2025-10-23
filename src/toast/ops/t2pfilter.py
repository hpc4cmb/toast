# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

import numpy as np
import traitlets
from astropy import units as u

from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Int, Unicode, trait_docs
from ..utils import Environment, Logger
from .operator import Operator


@trait_docs
class T2PFilter(Operator):
    """Operator that projects intensity out from demodulated data"""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key",
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for optional shared flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for detector sample flagging",
    )

    filter_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for flagging unfiltered samples",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        t0 = time()
        env = Environment.get()
        log = Logger.get()

        wcomm = data.comm.comm_world
        gcomm = data.comm.comm_group

        # Each process filters its data locally.
        # Demodulation leaves I/Q/U streams on the same process

        for ob in data.obs:
            # Cache the common flags
            if self.shared_flags is None:
                common_flags = np.zeros(ob.n_local_samples, dtype=np.uint8)
            else:
                common_flags = ob.shared[self.shared_flags].data & self.shared_flag_mask
            dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
            dets = set(dets)
            # Find the intensity streams
            prefix0 = "demod0"  # intensity stream prefix
            for det0 in dets:
                if not det0.startswith(prefix0):
                    # not an intensity stream
                    continue
                sigI = ob.detdata[self.det_data][det0]
                if self.det_flags is None:
                    flagI = np.zeros(ob.n_local_samples, dtype=np.uint8)
                else:
                    flagI = ob.detdata[self.det_flags][det0] & self.det_flag_mask
                for prefix in ["demod4r", "demod4i"]:
                    det = det0.replace(prefix0, prefix)
                    if det not in dets:
                        # The polarized stream is not available
                        continue
                    sig = ob.detdata[self.det_data][det]
                    if self.det_flags is None:
                        flag = np.zeros(ob.n_local_samples, dtype=np.uint8)
                    else:
                        flag = ob.detdata[self.det_flags][det]
                    good = (common_flags | flagI | (flag & self.det_flag_mask)) == 0
                    bad = np.logical_not(good)
                    # Project intensity out of polarization in each view
                    not_filtered = np.ones(ob.n_local_samples, dtype=bool)
                    for ival in ob.intervals[self.view]:
                        ind = slice(ival.first, ival.last)
                        # Build template matrix
                        templates = np.vstack(
                            [
                                np.ones(np.sum(good[ind])),
                                sigI[ind][good[ind]],
                            ]
                        )
                        # Get regression coefficients
                        invcov = np.dot(templates, templates.T)
                        try:
                            cov = np.linalg.inv(invcov)
                        except np.linalg.LinAlgError:
                            # Cannot filter this view
                            continue
                        proj = np.dot(templates, sig[ind][good[ind]])
                        # Subtract the templates
                        coeff = np.dot(cov, proj)
                        sig[ind] -= coeff[0] + coeff[1] * sigI[ind]
                        # Flag all samples that could not be used
                        flag[ind][bad[ind]] |= self.filter_flag_mask
                        not_filtered[ind] = False
                    # Raise flags for every unfiltered sample
                    flag[not_filtered] |= self.filter_flag_mask

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
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
        return dict()
