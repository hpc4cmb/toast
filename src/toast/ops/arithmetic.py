# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

from ..mpi import MPI
from ..timing import function_timer
from ..traits import Int, Unicode, trait_docs
from ..utils import Logger, unit_conversion
from .operator import Operator


@trait_docs
class Combine(Operator):
    """Arithmetic with detector data.

    Two detdata objects are combined element-wise using addition, subtraction,
    multiplication, or division.  The desired operation is specified by the "op"
    trait as a string.  The result is stored in the specified detdata object:

    result = first (op) second

    If the result name is the same as the first or second input, then this
    input will be overwritten.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    op = Unicode(
        None,
        allow_none=True,
        help="Operation on the timestreams: 'subtract', 'add', 'multiply', or 'divide'",
    )

    first = Unicode(None, allow_none=True, help="The first detdata object")

    second = Unicode(None, allow_none=True, help="The second detdata object")

    result = Unicode(None, allow_none=True, help="The resulting detdata object")

    @traitlets.validate("op")
    def _check_op(self, proposal):
        val = proposal["value"]
        if val is not None:
            if val not in ["add", "subtract", "multiply", "divide"]:
                raise traitlets.TraitError("op must be one of the 4 allowed strings")
        return val

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for check_name, check_val in [
            ("first", self.first),
            ("second", self.second),
            ("result", self.result),
            ("op", self.op),
        ]:
            if check_val is None:
                msg = f"The {check_name} trait must be set before calling exec"
                log.error(msg)
                raise RuntimeError(msg)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            if self.first not in ob.detdata:
                msg = "The first detdata key '{}' does not exist in observation {}".format(
                    self.first, ob.name
                )
                log.error(msg)
                raise RuntimeError(msg)
            if self.second not in ob.detdata:
                msg = "The second detdata key '{}' does not exist in observation {}".format(
                    self.second, ob.name
                )
                log.error(msg)
                raise RuntimeError(msg)

            first_units = ob.detdata[self.first].units
            second_units = ob.detdata[self.second].units

            if self.result == self.first:
                result_units = first_units
                scale_first = 1.0
                scale_second = unit_conversion(second_units, result_units)
            elif self.result == self.second:
                result_units = second_units
                scale_first = unit_conversion(first_units, result_units)
                scale_second = 1.0
            else:
                # We are creating a new field for the output.  Use units of first field.
                result_units = first_units
                scale_first = 1.0
                scale_second = unit_conversion(second_units, result_units)
                exists = ob.detdata.ensure(
                    self.result,
                    sample_shape=ob.detdata[self.first].detector_shape[1:],
                    dtype=ob.detdata[self.first].dtype,
                    detectors=ob.detdata[self.first].detectors,
                    create_units=result_units,
                )
            if self.op == "add":
                for d in dets:
                    ob.detdata[self.result][d, :] = (
                        scale_first * ob.detdata[self.first][d, :]
                    ) + (scale_second * ob.detdata[self.second][d, :])
            elif self.op == "subtract":
                for d in dets:
                    ob.detdata[self.result][d, :] = (
                        scale_first * ob.detdata[self.first][d, :]
                    ) - (scale_second * ob.detdata[self.second][d, :])
            elif self.op == "multiply":
                for d in dets:
                    ob.detdata[self.result][d, :] = (
                        scale_first * ob.detdata[self.first][d, :]
                    ) * (scale_second * ob.detdata[self.second][d, :])
            elif self.op == "divide":
                for d in dets:
                    ob.detdata[self.result][d, :] = (
                        scale_first * ob.detdata[self.first][d, :]
                    ) / (scale_second * ob.detdata[self.second][d, :])

    def _finalize(self, data, **kwargs):
        return None

    def _requires(self):
        req = {"detdata": [self.first, self.second]}
        if self.result is not None:
            if (self.result != self.first) and (self.result != self.second):
                req["detdata"].append(self.result)
        return req

    def _provides(self):
        prov = {"detdata": list()}
        if (self.result != self.first) and (self.result != self.second):
            prov["detdata"].append(self.result)
        return prov
