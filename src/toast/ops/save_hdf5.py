# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import traitlets

from ..io import save_hdf5
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Dict, Int, List, Unicode, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class SaveHDF5(Operator):
    """Operator which saves observations to HDF5.

    This creates a file for each observation.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    volume = Unicode("toast_out_hdf5", help="Top-level directory for the data volume")

    # FIXME:  We should add a filtering mechanism here to dump a subset of
    # observations and / or detectors, as well as figure out subdirectory organization.

    meta = List(None, allow_none=True, help="Only save this list of meta objects")

    detdata = List(None, allow_none=True, help="Only save this list of detdata objects")

    shared = List(None, allow_none=True, help="Only save this list of shared objects")

    intervals = List(
        None, allow_none=True, help="Only save this list of intervals objects"
    )

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    config = Dict(None, allow_none=True, help="Write this job config to the file")

    force_serial = Bool(
        False, help="Use serial HDF5 operations, even if parallel support available"
    )

    detdata_float32 = Bool(
        False, help="If True, convert any float64 detector data to float32 on write."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        # One process creates the top directory
        if data.comm.world_rank == 0:
            os.makedirs(self.volume, exist_ok=True)
        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        meta_fields = None
        if self.meta is not None and len(self.meta) > 0:
            meta_fields = list(self.meta)

        shared_fields = None
        if self.shared is not None and len(self.shared) > 0:
            shared_fields = list(self.shared)

        detdata_fields = None
        if self.detdata is not None and len(self.detdata) > 0:
            detdata_fields = list(self.detdata)

        intervals_fields = None
        if self.intervals is not None and len(self.intervals) > 0:
            intervals_fields = list(self.intervals)

        for ob in data.obs:
            # Observations must have a name for this to work
            if ob.name is None:
                raise RuntimeError(
                    "Observations must have a name in order to save to HDF5 format"
                )

            # Check to see if any detector data objects are temporary and have just
            # a partial list of detectors.  Delete these.

            for dd in list(ob.detdata.keys()):
                if ob.detdata[dd].detectors != ob.local_detectors:
                    del ob.detdata[dd]

            outpath = save_hdf5(
                ob,
                self.volume,
                meta=meta_fields,
                detdata=detdata_fields,
                shared=shared_fields,
                intervals=intervals_fields,
                config=self.config,
                times=str(self.times),
                force_serial=self.force_serial,
                detdata_float32=self.detdata_float32,
            )

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "detdata": list(),
            "intervals": list(),
            "shared": [
                self.times,
            ],
        }
        if self.meta is not None:
            req["meta"].extend(self.meta)
        if self.detdata is not None:
            req["detdata"].extend(self.detdata)
        if self.intervals is not None:
            req["intervals"].extend(self.intervals)
        if self.shared is not None:
            req["shared"].extend(self.shared)
        return req

    def _provides(self):
        return dict()
