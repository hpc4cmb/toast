# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numbers

import traitlets

from ..timing import function_timer
from ..traits import Int, List, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class Reset(Operator):
    """Class to reset data from observations.

    This operator takes lists of shared, detdata, intervals, and meta keys to reset.
    Numerical data objects and arrays are set to zero.  String objects are set to an
    empty string.  Any object that defines a `clear()` method will have that called.
    Any object not matching those criteria will be set to None.  Since an IntervalList
    is not mutable, any specified intervals will simply be deleted.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    meta = List([], help="List of Observation dictionary keys to reset")

    detdata = List([], help="List of Observation detdata keys to reset")

    shared = List([], help="List of Observation shared keys to reset")

    intervals = List(
        [],
        help="List of tuples of Observation intervals keys to reset",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        for ob in data.obs:
            if len(self.detdata) > 0:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                for key in self.detdata:
                    for d in dets:
                        ob.detdata[key][d, :] = 0
            for key in self.shared:
                scomm = ob.shared[key].nodecomm
                if scomm is None:
                    # No MPI, just set to zero
                    ob.shared[key].data[:] = 0
                else:
                    # Only rank zero on each node resets
                    if scomm.rank == 0:
                        ob.shared[key]._flat[:] = 0
                    scomm.barrier()
            for key in self.intervals:
                # This ignores non-existant keys
                del ob.intervals[key]
            for key in self.meta:
                if isinstance(ob[key], np.ndarray):
                    # This is an array, set to zero
                    ob[key][:] = 0
                elif hasattr(ob[key], "clear"):
                    # This is some kind of container (list, dict, etc).  Clear it.
                    ob[key].clear()
                elif isinstance(ob[key], bool):
                    # Boolean scalar, set to False
                    ob[key] = False
                elif isinstance(ob[key], numbers.Number):
                    # This is a scalar numeric value
                    ob[key] = 0
                elif isinstance(ob[key], (str, bytes)):
                    # This is string-like
                    ob[key] = ""
                else:
                    # This is something else.  Set to None
                    ob[key] = None
        return

    def _finalize(self, data, **kwargs):
        return None

    def _requires(self):
        req = dict()
        if self.meta is not None:
            req["meta"] = list(self.meta)
        if self.detdata is not None:
            req["detdata"] = list(self.detdata)
        if self.shared is not None:
            req["shared"] = list(self.shared)
        if self.intervals is not None:
            req["intervals"] = list(self.intervals)
        return req

    def _provides(self):
        return dict()
