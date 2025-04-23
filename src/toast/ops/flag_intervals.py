# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets

from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Int, List, Unicode, trait_docs, Bool
from ..utils import Environment, Logger
from .operator import Operator


@trait_docs
class FlagIntervals(Operator):
    """Operator which updates shared flags from interval lists.

    This operator can be used in cases where interval information needs to be combined
    with shared flags.  The view_mask trait is a list of tuples.  Each tuple contains
    the name of the view (i.e. interval) to apply and the bitmask to use for that
    view.  For each interval view, flag values in the shared_flags object are bitwise-
    OR'd with the specified mask for samples in the view.  If the name of the view is
    prefixed with '~' the bitmask is applied to all samples outside the view.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    view_mask = List([], help="List of tuples of (view name, bit mask)")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_bytes = Int(
        1, help="If creating shared key, use this many bytes per sample"
    )

    reset = Bool(
        False, help="If True, flag bits are first set to 0 for the entire observation"
    )

    @traitlets.validate("shared_flag_bytes")
    def _check_flag_bytes(self, proposal):
        check = proposal["value"]
        if check not in [1, 2, 4, 8]:
            raise traitlets.TraitError("shared flag byte width should be 1, 2, 4, or 8")
        return check

    @traitlets.validate("view_mask")
    def _check_flag_mask(self, proposal):
        check = proposal["value"]
        for vname, vmask in check:
            if vmask < 0:
                raise traitlets.TraitError("Flag masks should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.shared_flags is None:
            log.debug_rank(
                "shared_flags trait is None, nothing to do.", comm=data.comm.comm_world
            )
            return

        if self.view_mask is None or len(self.view_mask) == 0:
            log.debug_rank(
                "view_mask trait is empty or not set, nothing to do.",
                comm=data.comm.world_comm,
            )
            return

        fdtype = None
        if self.shared_flag_bytes == 8:
            fdtype = np.uint64
        elif self.shared_flag_bytes == 4:
            fdtype = np.uint32
        elif self.shared_flag_bytes == 2:
            fdtype = np.uint16
        else:
            fdtype = np.uint8

        for ob in data.obs:
            # If the shared flag object already exists, then use it with whatever
            # byte width is in place.  Otherwise create it.

            if self.shared_flags not in ob.shared:
                ob.shared.create_column(
                    self.shared_flags,
                    shape=(ob.n_local_samples,),
                    dtype=fdtype,
                )

            # The intervals / view is common between all processes in a column of the
            # process grid.  Only the rank zero process in each column builds the new
            # flags for the synchronous call to the set() method.  Note that views
            # of shared data are read-only, so we build the full flag vector and only
            # modify samples inside the view.

            new_flags = None
            if ob.comm_col_rank == 0:
                new_flags = np.array(ob.shared[self.shared_flags])
                if self.reset:
                    for vname, vmask in self.view_mask:
                        new_flags &= ~vmask
                for vname, vmask in self.view_mask:
                    try:
                        for vw in ob.view[vname]:
                            # Note that a View acts like a slice
                            new_flags[vw] |= vmask
                    except KeyError as e:
                        msg = f"{e}; Intervals '{vname}' does not exist in {ob.name}"
                        msg += " skipping flagging"
                        log.warning(msg)
            ob.shared[self.shared_flags].set(new_flags, offset=(0,), fromrank=0)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        if self.view_mask is not None:
            req["intervals"] = [x[0] for x in self.view_mask]
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": [self.shared_flags],
            "detdata": list(),
            "intervals": list(),
        }
        return prov
