# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import OrderedDict

import traitlets

from ..utils import Logger

from ..timing import function_timer

from ..traits import trait_docs, Int, Unicode, List

from ..data import Data

from .operator import Operator


@trait_docs
class Pipeline(Operator):
    """Class representing a sequence of Operators.

    This runs a list of other operators over sets of detectors (default is all
    detectors in one shot).  By default all observations are passed to each operator,
    but the `observation_key` and `observation_value` traits can be used to run the
    operators on only observations which have a matching key / value pair.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    operators = List(allow_none=True, help="List of Operator instances to run.")

    detector_sets = List(
        ["ALL"],
        help="List of detector sets.  ['ALL'] and ['SINGLE'] are also valid values.",
    )

    @traitlets.validate("detector_sets")
    def _check_detsets(self, proposal):
        detsets = proposal["value"]
        if len(detsets) == 0:
            raise traitlets.TraitError(
                "detector_sets must be a list with at least one entry ('ALL' and 'SINGLE' are valid entries)"
            )
        for dset in detsets:
            if (dset != "ALL") and (dset != "SINGLE"):
                # Not a built-in name, must be an actual list of detectors
                if isinstance(dset, str) or len(dset) == 0:
                    raise traitlets.TraitError(
                        "A detector set must be a list of detectors or 'ALL' / 'SINGLE'"
                    )
                for d in dset:
                    if not isinstance(d, str):
                        raise traitlets.TraitError(
                            "Each element of a det set should be a detector name"
                        )
        return detsets

    @traitlets.validate("operators")
    def _check_operators(self, proposal):
        ops = proposal["value"]
        if ops is None:
            return ops
        for op in ops:
            if not isinstance(op, Operator):
                raise traitlets.TraitError(
                    "operators must be a list of Operator instances or None"
                )
        return ops

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        pstr = f"Proc ({data.comm.world_rank}, {data.comm.group_rank})"

        acc = self.accelerators()

        if "CUDA" in acc:
            # All our operators support CUDA.  Stage any required data
            pass

        if len(data.obs) == 0:
            # No observations for this group
            msg = f"Pipeline data, group {data.comm.group} has no observations."
            log.verbose_rank(msg, comm=data.comm.comm_group)

        if len(self.detector_sets) == 1 and self.detector_sets[0] == "ALL":
            # Run the operators with all detectors at once
            for op in self.operators:
                msg = "{} Pipeline calling operator '{}' exec() with ALL dets".format(
                    pstr, op.name
                )
                log.verbose(msg)
                op.exec(data, detectors=None)
        elif len(self.detector_sets) == 1 and self.detector_sets[0] == "SINGLE":
            # Get superset of detectors across all observations
            all_local_dets = data.all_local_detectors(selection=detectors)
            # Run operators one detector at a time
            for det in all_local_dets:
                msg = "{} Pipeline SINGLE detector {}".format(pstr, det)
                log.verbose(msg)
                for op in self.operators:
                    msg = "{} Pipeline   calling operator '{}' exec()".format(
                        pstr, op.name
                    )
                    log.verbose(msg)
                    op.exec(data, detectors=[det])
        else:
            # We have explicit detector sets
            det_check = set(detectors)
            for det_set in self.detector_sets:
                selected_set = det_set
                if detectors is not None:
                    selected_set = list()
                    for det in det_set:
                        if det in det_check:
                            selected_set.append(det)
                if len(selected_set) == 0:
                    # Nothing in this detector set is being used, skip it
                    continue
                msg = "{} Pipeline detector set {}".format(pstr, selected_set)
                log.verbose(msg)
                for op in self.operators:
                    msg = "{} Pipeline   calling operator '{}' exec()".format(
                        pstr, op.name
                    )
                    log.verbose(msg)
                    op.exec(data, detectors=selected_set)

        # Copy from accelerator...

        return

    @function_timer
    def _finalize(self, data, **kwargs):
        log = Logger.get()
        result = list()
        pstr = f"Proc ({data.comm.world_rank}, {data.comm.group_rank})"
        if self.operators is not None:
            for op in self.operators:
                msg = f"{pstr} Pipeline calling operator '{op.name}' finalize()"
                log.verbose(msg)
                result.append(op.finalize(data))
        return result

    def _requires(self):
        # Work through the operator list in reverse order and prune intermediate
        # products.
        if self.operators is None:
            return dict()
        keys = ["meta", "detdata", "shared", "intervals"]
        req = {x: set() for x in keys}
        for op in reverse(self.operators):
            oreq = op.requires()
            oprov = op.provides()
            for k in keys:
                if k in oreq:
                    req[k] |= oreq[k]
                if k in oprov:
                    req[k] -= oprov[k]
        for k in keys:
            req[k] = list(req[k])
        return req

    def _provides(self):
        # Work through the operator list and prune intermediate products.
        if self.operators is None:
            return dict()
        keys = ["meta", "detdata", "shared", "intervals"]
        prov = {x: set() for x in keys}
        for op in self.operators:
            oreq = op.requires()
            oprov = op.provides()
            for k in keys:
                if k in oprov:
                    prov[k] |= oprov[k]
                if k in oreq:
                    prov[k] -= oreq[k]
        for k in keys:
            prov[k] = list(prov[k])
        return prov

    def _accelerators(self):
        # This is just the intersection of results from all operators in our list.
        if self.operators is None:
            return list()
        acc = set()
        for op in self.operators:
            for support in op.accelerators():
                acc.add(support)
        for op in self.operators:
            supported = op.accelerators()
            for a in list(acc):
                if a not in supported:
                    acc.remove(a)
        return list(acc)
