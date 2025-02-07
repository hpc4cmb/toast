# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets
import numpy as np
from astropy import units as u

from ..mpi import MPI
from ..timing import function_timer
from ..traits import Int, List, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class Create(Operator):
    """Class to pre-create detector and shared data.

    Many operators already create data objects if they do not exist, but sometimes it
    is necessary to pre-create data objects as part of a Pipeline.

    Each list contains tuples of properties of the field to create.

    Each shared tuple should contain the (name, commtype, shape, dtype) where commtype
    may be "group", "row", or "column".  dtype should be the string name ("float64",
    "int32", "uint8", etc).

    Each detector data tuple should contain the (name, sample_shape, dtype, units) where
    dtype and units are a string.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    detdata = List([], help="List of tuples of Observation detdata keys to create")

    shared = List([], help="List of tuples of Observation shared keys to create")

    @traitlets.validate("detdata")
    def _check_detdata(self, proposal):
        val = proposal["value"]
        for v in val:
            if not isinstance(v, (tuple, list)):
                raise traitlets.TraitError("trait should be a list of tuples")
            if len(v) != 4:
                raise traitlets.TraitError("key tuples should have 4 values")
            if not isinstance(v[0], str) or not isinstance(v[1], str):
                raise traitlets.TraitError("key tuples should have string values")
        return val

    @traitlets.validate("shared")
    def _check_shared(self, proposal):
        val = proposal["value"]
        for v in val:
            if not isinstance(v, (tuple, list)):
                raise traitlets.TraitError("trait should be a list of tuples")
            if len(v) != 4:
                raise traitlets.TraitError("key tuples should have 4 values")
            if not isinstance(v[0], str) or not isinstance(v[1], str):
                raise traitlets.TraitError("key tuples should have string values")
        return val

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _dtype_from_str(self, dstr):
        if dstr == "int8":
            return np.dtype(np.int8)
        elif dstr == "uint8":
            return np.dtype(np.uint8)
        elif dstr == "int16":
            return np.dtype(np.int16)
        elif dstr == "uint16":
            return np.dtype(np.uint16)
        elif dstr == "int32":
            return np.dtype(np.int32)
        elif dstr == "uint32":
            return np.dtype(np.uint32)
        elif dstr == "int64":
            return np.dtype(np.int64)
        elif dstr == "uint64":
            return np.dtype(np.uint64)
        elif dstr == "float32":
            return np.dtype(np.float32)
        elif dstr == "float64":
            return np.dtype(np.float64)
        else:
            msg = f"Unsupported dtype '{dstr}'"
            raise RuntimeError(msg)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        det_props = list()
        for dp in self.detdata:
            if len(dp) != 4:
                msg = f"Expected 4 elements in detdata field spec, got '{dp}'"
                raise ValueError(msg)
            dname = str(dp[0])
            if isinstance(dp[1], tuple):
                dshp = dp[1]
            else:
                dshp = eval(dp[1])
            dt = self._dtype_from_str(dp[2])
            if isinstance(dp[3], u.Unit):
                dunit = dp[3]
            else:
                dunit = u.Unit(dp[3])
            det_props.append((dname, dshp, dt, dunit))
        shared_props = list()
        for sp in self.shared:
            sname = str(sp[0])
            scomm = str(sp[1])
            if isinstance(sp[2], tuple):
                sshp = sp[2]
            else:
                sshp = eval(sp[2])
            dt = self._dtype_from_str(sp[3])
            shared_props.append((sname, scomm, sshp, dt))

        for ob in data.obs:
            for sprops in shared_props:
                if sprops[0] in ob.shared:
                    # Shared field exists, check for consistency
                    cur_type = ob.shared.comm_type(sprops[0])
                    if cur_type != sprops[1]:
                        msg = f"Existing shared field '{sprops[0]}' has different "
                        msg += f"communicator ({cur_type}) than requested ({sprops[1]})"
                        raise RuntimeError(msg)
                    cur_shp = ob.shared[sprops[0]].shdata.shape
                    if cur_shp != sprops[2]:
                        msg = f"Existing shared field '{sprops[0]}' has different "
                        msg += f"shape ({cur_shp}) than requested ({sprops[2]})"
                        raise RuntimeError(msg)
                    cur_dt = ob.shared[sprops[0]].shdata.dtype
                    if cur_dt != sprops[3]:
                        msg = f"Existing shared field '{sprops[0]}' has different "
                        msg += f"dtype ({cur_dt}) than requested ({sprops[3]})"
                        raise RuntimeError(msg)
                else:
                    # Create it
                    ob.shared.create_type(
                        sprops[1], sprops[0], sprops[2], dtype=sprops[3]
                    )

            for dprops in det_props:
                exists = ob.detdata.ensure(
                    dprops[0],
                    sample_shape=dprops[1],
                    dtype=dprops[2],
                    create_units=dprops[3],
                )

        return

    def _finalize(self, data, **kwargs):
        return None

    def _requires(self):
        req = dict()
        return req

    def _provides(self):
        prov = dict()
        prov["detdata"] = [x[0] for x in self.detdata]
        prov["shared"] = [x[0] for x in self.shared]
        return prov
