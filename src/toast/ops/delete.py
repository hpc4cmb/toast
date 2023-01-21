# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

from ..timing import function_timer
from ..traits import Int, List, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class Delete(Operator):
    """Class to purge data from observations.

    This operator takes lists of shared, detdata, intervals and meta keys to delete from
    observations.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    meta = List([], help="List of Observation dictionary keys to delete")

    detdata = List([], help="List of Observation detdata keys to delete")

    shared = List([], help="List of Observation shared keys to delete")

    intervals = List(
        [],
        help="List of tuples of Observation intervals keys to delete",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        for ob in data.obs:
            for key in self.detdata:
                # This ignores non-existant keys
                del ob.detdata[key]
            for key in self.shared:
                # This ignores non-existant keys
                del ob.shared[key]
            for key in self.intervals:
                # This ignores non-existant keys
                del ob.intervals[key]
            for key in self.meta:
                try:
                    del ob[key]
                except KeyError:
                    pass
        return

    def _finalize(self, data, **kwargs):
        return None

    def _requires(self):
        # Although we could require nothing, since we are deleting keys only if they
        # exist, providing these as requirements allows us to catch dependency issues
        # in pipelines.
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
