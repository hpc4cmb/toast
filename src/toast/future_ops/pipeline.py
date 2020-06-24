# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, List

from ..operator import Operator


class Pipeline(Operator):
    """Class representing a sequence of Operators."""

    # Class traits

    API = traitlets.Int(0, help="Internal interface version for this operator")

    operators = List(allow_none=True, help="List of Operator instances to run.")

    detector_sets = List(
        ["ALL"],
        help="List of detector sets.  'ALL' and 'SINGLE' are also valid values.",
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

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        acc = self.accelerators()

        if "CUDA" in acc:
            # All our operators support CUDA.  Stage any required data
            pass

        if detectors is not None:
            msg = "Use the 'detector_sets' option to control a Pipeline"
            log.error(msg)
            raise RuntimeError(msg)

        for dset in self.detector_sets:
            if dset == "ALL":
                for op in self.operators:
                    op.exec(data)
            elif dset == "SINGLE":
                # We are running one detector at a time
                raise NotImplementedError("SINGLE detectors not implemented yet")
            else:
                # We are running sets of detectors at once.  We first go through all
                # observations and find the set of detectors used by each row of the
                # process grid.
                raise NotImplementedError("detector sets not implemented yet")

        # Copy to / from accelerator...

        return

    def _finalize(self, data, **kwargs):
        if self.operators is not None:
            for op in self.operators:
                op.finalize(data)

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
                req[k] |= oreq[k]
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
                prov[k] |= oprov[k]
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
