# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import OrderedDict

import traitlets

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, List

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

    observation_key = Unicode(
        None,
        allow_none=True,
        help="Only process observations which have this key defined",
    )

    observation_value = Unicode(
        None,
        allow_none=True,
        help="Only process observations where the key has this value",
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

    @traitlets.validate("observation_value")
    def _check_observation_value(self, proposal):
        val = proposal["value"]
        if val is None:
            return val
        if self.observation_key is None:
            raise traitlets.TraitError(
                "observation_key must be set before observation_value"
            )
        return val

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        pstr = f"Proc ({data.comm.world_rank}, {data.comm.group_rank})"

        acc = self.accelerators()

        if "CUDA" in acc:
            # All our operators support CUDA.  Stage any required data
            pass

        # Select the observations we will use
        data_sets = [data]
        if self.observation_key is not None:
            data_sets = list()
            split_data = data.split(self.observation_key)
            if self.observation_value is None:
                # We are using all values of the key
                for val, d in split_data.items():
                    data_sets.append(d)
            else:
                # We are using only one value of the key
                if self.observation_value not in split_data:
                    msg = "input data has no observations where '{}' == '{}'".format(
                        self.observation_key, self.observation_value
                    )
                    if data.comm.world_rank == 0:
                        log.warning(msg)
                else:
                    data_sets.append(split_data[self.observation_value])

        for ds_indx, ds in enumerate(data_sets):
            if len(ds.obs) == 0:
                # No observations for this group
                msg = "Pipeline starting data set {}, group {} has no observations.\n".format(
                    ds_indx, ds.comm.group
                )
                if data.comm.group_rank == 0:
                    log.verbose(msg)
            for det_set in self.detector_sets:
                if det_set == "ALL":
                    # If this is given, then there should be only one entry
                    if len(self.detector_sets) != 1:
                        raise RuntimeError(
                            "If using 'ALL' for a detector set, there should only be one set"
                        )
                    # The superset of detectors across all observations.
                    all_local_dets = OrderedDict()
                    for ob in ds.obs:
                        for det in ob.local_detectors:
                            all_local_dets[det] = None
                    all_local_dets = list(all_local_dets.keys())

                    # If we have no detectors at all across all observations, it
                    # means that one of our operators is going to create observations.
                    # pass None for the list of detectors.
                    selected_dets = None
                    if len(all_local_dets) > 0:
                        selected_dets = all_local_dets
                        # If we were given a more restrictive list, prune the global
                        # list
                        if detectors is not None:
                            selected_dets = list()
                            for det in all_local_dets:
                                if det in detectors:
                                    selected_dets.append(det)

                    # Run the operators with this full list
                    for op in self.operators:
                        msg = "{} Pipeline calling operator '{}' exec() with ALL dets".format(
                            pstr, op.name
                        )
                        log.verbose(msg)
                        op.exec(ds, detectors=selected_dets)
                elif det_set == "SINGLE":
                    # If this is given, then there should be only one entry
                    if len(self.detector_sets) != 1:
                        raise RuntimeError(
                            "If using 'SINGLE' for a detector set, there should only be one set"
                        )

                    # We are running one detector at a time.  We will loop over all
                    # detectors in the superset of detectors across all observations.
                    all_local_dets = OrderedDict()
                    for ob in ds.obs:
                        for det in ob.local_detectors:
                            all_local_dets[det] = None
                    all_local_dets = list(all_local_dets.keys())

                    # If we were given a more restrictive list, prune the global list
                    selected_dets = all_local_dets
                    if detectors is not None:
                        selected_dets = list()
                        for det in all_local_dets:
                            if det in detectors:
                                selected_dets.append(det)

                    for det in selected_dets:
                        msg = "{} Pipeline SINGLE detector {}".format(pstr, det)
                        log.verbose(msg)
                        for op in self.operators:
                            msg = "{} Pipeline   calling operator '{}' exec()".format(
                                pstr, op.name
                            )
                            log.verbose(msg)
                            op.exec(ds, detectors=[det])
                else:
                    # We are running sets of detectors at once.  For this detector
                    # set, we prune to just the restricted list passed to exec().
                    selected_set = det_set
                    if detectors is not None:
                        selected_set = list()
                        for det in det_set:
                            if det in detectors:
                                selected_set.append(det)
                    msg = "{} Pipeline detector set {}".format(pstr, selected_set)
                    log.verbose(msg)
                    for op in self.operators:
                        msg = "{} Pipeline   calling operator '{}' exec()".format(
                            pstr, op.name
                        )
                        log.verbose(msg)
                        op.exec(ds, detectors=selected_set)

        # Copy to / from accelerator...

        return

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
