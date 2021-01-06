# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

from ..utils import Logger

from ..mpi import MPI

from ..traits import trait_docs, Int, Unicode, List

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

    meta = List(
        None, allow_none=True, help="List of tuples of Observation meta keys to copy"
    )

    detdata = List(
        None, allow_none=True, help="List of tuples of Observation detdata keys to copy"
    )

    shared = List(
        None, allow_none=True, help="List of tuples of Observation shared keys to copy"
    )

    @traitlets.validate("meta")
    def _check_meta(self, proposal):
        val = proposal["value"]
        if val is None:
            return val
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
        if val is None:
            return val
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
        if val is None:
            return val
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

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        for ob in data.obs:
            if self.meta is not None:
                for in_key, out_key in self.meta:
                    if out_key in ob:
                        # The key exists- issue a warning before overwriting.
                        msg = "Observation key {} already exists- overwriting".format(
                            out_key
                        )
                        log.warning(msg)
                    ob[out_key] = ob[in_key]

            if self.shared is not None:
                for in_key, out_key in self.shared:
                    if out_key in ob.shared:
                        if ob.shared[in_key].comm is not None:
                            comp = MPI.Comm.Compare(
                                ob.shared[out_key].comm, ob.shared[in_key].comm
                            )
                            if comp not in (MPI.IDENT, MPI.CONGRUENT):
                                msg = "Cannot copy to existing shared key {} with a different communicator".format(
                                    out_key
                                )
                                log.error(msg)
                                raise RuntimeError(msg)
                        if ob.shared[out_key].dtype != ob.shared[in_key].dtype:
                            msg = "Cannot copy to existing shared key {} with different dtype".format(
                                out_key
                            )
                            log.error(msg)
                            raise RuntimeError(msg)
                        if ob.shared[out_key].shape != ob.shared[in_key].shape:
                            msg = "Cannot copy to existing shared key {} with different shape".format(
                                out_key
                            )
                            log.error(msg)
                            raise RuntimeError(msg)
                    else:
                        ob.shared.create(
                            out_key,
                            shape=ob.shared[in_key].shape,
                            dtype=ob.shared[in_key].dtype,
                            comm=ob.shared[in_key].comm,
                        )
                    # Only one process per node copies the shared data.
                    if (
                        ob.shared[in_key].nodecomm is None
                        or ob.shared[in_key].nodecomm.rank == 0
                    ):
                        ob.shared[out_key]._flat[:] = ob.shared[in_key]._flat

            if self.detdata is not None:
                for in_key, out_key in self.detdata:
                    if out_key in ob.detdata:
                        # The key exists- verify that dimensions match
                        if (
                            ob.detdata[out_key].detectors
                            != ob.detdata[in_key].detectors
                        ):
                            msg = "Cannot copy to existing detdata key {} with different detectors".format(
                                out_key
                            )
                            log.error(msg)
                            raise RuntimeError(msg)
                        if ob.detdata[out_key].dtype != ob.detdata[in_key].dtype:
                            msg = "Cannot copy to existing detdata key {} with different dtype".format(
                                out_key
                            )
                            log.error(msg)
                            raise RuntimeError(msg)
                        if ob.detdata[out_key].shape != ob.detdata[in_key].shape:
                            msg = "Cannot copy to existing detdata key {} with different shape".format(
                                out_key
                            )
                            log.error(msg)
                            raise RuntimeError(msg)
                    else:
                        sample_shape = None
                        shp = ob.detdata[in_key].detector_shape
                        if len(shp) > 1:
                            sample_shape = shp[1:]
                        ob.detdata.create(
                            out_key,
                            sample_shape=sample_shape,
                            dtype=ob.detdata[in_key].dtype,
                            detectors=ob.detdata[in_key].detectors,
                        )
                    ob.detdata[out_key][:] = ob.detdata[in_key][:]

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
        return req

    def _provides(self):
        prov = dict()
        if self.meta is not None:
            req["meta"] = [x[1] for x in self.meta]
        if self.detdata is not None:
            req["detdata"] = [x[1] for x in self.detdata]
        if self.shared is not None:
            req["shared"] = [x[1] for x in self.shared]
        return prov

    def _accelerators(self):
        # Eventually we can copy memory objects on devices...
        return list()
