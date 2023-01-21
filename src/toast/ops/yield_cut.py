# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from .. import rng
from .._libtoast import add_templates, bin_invcov, bin_proj, legendre
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Float, Int, Unicode, trait_docs
from ..utils import Environment, Logger, Timer, name_UID
from .operator import Operator


@trait_docs
class YieldCut(Operator):
    """Operator that simulates non-perfect yield.

    When TES detectors have their bias tuned, not all detectors have sufficient
    responsivity to be useful for science.  This can be a temporary problem.  This
    operator simulates a random loss in detector yield.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    keep_frac = Float(0.9, help="Fraction of detectors to keep")

    focalplane_key = Unicode(
        "pixel",
        help="Which focalplane key to use for randomization.  "
        "Detectors that share the key value are flagged together",
    )

    fixed = Bool(
        False,
        help="If True, detector cuts do not change between observations "
        "and realizations",
    )

    realization = Int(0, help="The realization index")

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        t0 = time()
        env = Environment.get()
        log = Logger.get()

        for obs in data.obs:
            focalplane = obs.telescope.focalplane
            dets = obs.select_local_detectors(detectors)

            # For reproducibility, generate the random cut across all detectors
            # in the observation.  This means that a given process might have more
            # or less detectors cut than the target fraction, but the overall
            # value should be close for a large enough number of detectors.

            exists = obs.detdata.ensure(self.det_flags, dtype=np.uint8, detectors=dets)
            for det in dets:
                key1 = obs.telescope.uid
                if self.fixed:
                    key2 = 0
                    counter1 = 0
                else:
                    key2 = self.realization
                    counter1 = obs.session.uid
                if self.focalplane_key is not None:
                    value = focalplane[det][self.focalplane_key]
                    counter2 = name_UID(value)
                else:
                    counter2 = focalplane[det]["UID"]
                x = rng.random(
                    1,
                    sampler="uniform_01",
                    key=(key1, key2),
                    counter=(counter1, counter2),
                )[0]
                if x > self.keep_frac:
                    obs.detdata[self.det_flags][det] |= self.det_flag_mask
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_flags],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov
