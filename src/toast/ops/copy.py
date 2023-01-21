# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

from ..mpi import MPI
from ..timing import function_timer
from ..traits import Int, List, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class Copy(Operator):
    """Class to copy data.

    This operator takes lists of shared, detdata, and meta keys to copy to a new
    location in each observation.

    Each list contains tuples specifying the input and output key names.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    meta = List([], help="List of tuples of Observation meta keys to copy")

    detdata = List([], help="List of tuples of Observation detdata keys to copy")

    shared = List([], help="List of tuples of Observation shared keys to copy")

    intervals = List(
        [],
        help="List of tuples of Observation intervals keys to copy",
    )

    @traitlets.validate("meta")
    def _check_meta(self, proposal):
        val = proposal["value"]
        for v in val:
            if not isinstance(v, (tuple, list)):
                raise traitlets.TraitError("trait should be a list of tuples")
            if len(v) != 2:
                raise traitlets.TraitError("key tuples should have 2 values")
            if not isinstance(v[0], str) or not isinstance(v[1], str):
                raise traitlets.TraitError("key tuples should have string values")
        return val

    @traitlets.validate("detdata")
    def _check_detdata(self, proposal):
        val = proposal["value"]
        for v in val:
            if not isinstance(v, (tuple, list)):
                raise traitlets.TraitError("trait should be a list of tuples")
            if len(v) != 2:
                raise traitlets.TraitError("key tuples should have 2 values")
            if not isinstance(v[0], str) or not isinstance(v[1], str):
                raise traitlets.TraitError("key tuples should have string values")
        return val

    @traitlets.validate("shared")
    def _check_shared(self, proposal):
        val = proposal["value"]
        for v in val:
            if not isinstance(v, (tuple, list)):
                raise traitlets.TraitError("trait should be a list of tuples")
            if len(v) != 2:
                raise traitlets.TraitError("key tuples should have 2 values")
            if not isinstance(v[0], str) or not isinstance(v[1], str):
                raise traitlets.TraitError("key tuples should have string values")
        return val

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        for ob in data.obs:
            for in_key, out_key in self.meta:
                if out_key in ob:
                    # The key exists- issue a warning before overwriting.
                    msg = "Observation key {} already exists- overwriting".format(
                        out_key
                    )
                    log.warning(msg)
                ob[out_key] = ob[in_key]

            for in_key, out_key in self.shared:
                # Although this is an internal function, the input arguments come
                # from existing shared objects and so should already be valid.
                ob.shared.assign_mpishared(
                    out_key, ob.shared[in_key], ob.shared.comm_type(in_key)
                )

            if len(self.detdata) > 0:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                for in_key, out_key in self.detdata:
                    if out_key in ob.detdata:
                        # The key exists- verify that dimensions / dtype match
                        in_dtype = ob.detdata[in_key].dtype
                        out_dtype = ob.detdata[out_key].dtype
                        if out_dtype != in_dtype:
                            msg = f"Cannot copy to existing detdata key {out_key}"
                            msg += f" with different dtype ({out_dtype}) != {in_dtype}"
                            log.error(msg)
                            raise RuntimeError(msg)
                        in_shape = ob.detdata[in_key].detector_shape
                        out_shape = ob.detdata[out_key].detector_shape
                        if out_shape != in_shape:
                            msg = f"Cannot copy to existing detdata key {out_key}"
                            msg += f" with different detector shape ({out_shape})"
                            msg += f" != {in_shape}"
                            log.error(msg)
                            raise RuntimeError(msg)
                        if ob.detdata[out_key].detectors != dets:
                            # The output has a different set of detectors.  Reallocate.
                            ob.detdata[out_key].change_detectors(dets)
                        # Copy units
                        ob.detdata[out_key].update_units(ob.detdata[in_key].units)
                    else:
                        sample_shape = None
                        shp = ob.detdata[in_key].detector_shape
                        if len(shp) > 1:
                            sample_shape = shp[1:]
                        ob.detdata.create(
                            out_key,
                            sample_shape=sample_shape,
                            dtype=ob.detdata[in_key].dtype,
                            detectors=dets,
                            units=ob.detdata[in_key].units,
                        )
                    # Copy detector data
                    for d in dets:
                        ob.detdata[out_key][d, :] = ob.detdata[in_key][d, :]
        return

    def _finalize(self, data, **kwargs):
        return None

    def _requires(self):
        req = dict()
        if self.meta is not None:
            req["meta"] = [x[0] for x in self.meta]
        if self.detdata is not None:
            req["detdata"] = [x[0] for x in self.detdata]
        if self.shared is not None:
            req["shared"] = [x[0] for x in self.shared]
        if self.intervals is not None:
            req["intervals"] = [x[0] for x in self.intervals]
        return req

    def _provides(self):
        prov = dict()
        if self.meta is not None:
            prov["meta"] = [x[1] for x in self.meta]
        if self.detdata is not None:
            prov["detdata"] = [x[1] for x in self.detdata]
        if self.shared is not None:
            prov["shared"] = [x[1] for x in self.shared]
        if self.intervals is not None:
            prov["intervals"] = [x[1] for x in self.intervals]
        return prov
