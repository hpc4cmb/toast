# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import OrderedDict

import traitlets

from ..accelerator import ImplementationType, accel_enabled
from ..data import Data
from ..timing import function_timer
from ..traits import Int, List, trait_docs
from ..utils import Logger
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

    operators = List([], help="List of Operator instances to run.")

    detector_sets = List(
        ["ALL"],
        help="List of detector sets.  ['ALL'] and ['SINGLE'] are also valid values.",
    )

    @traitlets.validate("detector_sets")
    def _check_detsets(self, proposal):
        detsets = proposal["value"]
        if len(detsets) == 0:
            msg = "detector_sets must be a list with at least one entry "
            msg += "('ALL' and 'SINGLE' are valid entries)"
            raise traitlets.TraitError(msg)
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
        for op in ops:
            if not isinstance(op, Operator):
                raise traitlets.TraitError(
                    "operators must be a list of Operator instances or None"
                )
        return ops

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._staged_accel = False

    @function_timer
    def _exec(self, data, detectors=None, use_accel=False, **kwargs):
        log = Logger.get()

        pstr = f"Proc ({data.comm.world_rank}, {data.comm.group_rank})"

        if len(self.operators) == 0:
            log.debug_rank(
                "Pipeline has no operators, nothing to do", comm=data.comm.comm_world
            )
            return
        
        # There are 2 scenarios we handle here:  If the use_accel trait has already
        # been set to True, it means that the calling code has already queried the
        # supports_accel() method and it returned True, and the calling code has
        # ensured that all requirements are staged to the accelerator.  If the
        # use_accel trait is False, then we still may be able to run on the accelerator
        # if our operators support it and if we stage the data ourselves.

        # This is used to track whether we are handling the data staging internally.
        # If this is set to True in the finalize method, then we copy data back out
        # and restore our state.
        self._staged_accel = False

        if not use_accel:
            # The calling code determined that we do not have all the data present to
            # use the accelerator.  However, if all of our operators support it, we 
            # can stage the data to and from the device.
            if detectors is None:
                comp_dets = set(data.all_local_detectors(selection=None))
            else:
                comp_dets = set(detectors)
            if accel_enabled() and self.supports_accel():
                # All our operators support it.
                msg = f"{self} fully supports accelerators, data to "
                msg += f"be staged: {self.requires()}"
                log.verbose_rank(msg, comm=data.comm.comm_world)
                
                # Send the requirements to the device
                self._stage_requirements_to_device(data)
                self._staged_accel = True
                use_accel = True

        if len(data.obs) == 0:
            # No observations for this group
            msg = f"{self} data, group {data.comm.group} has no observations."
            log.verbose_rank(msg, comm=data.comm.comm_group)

        if len(self.detector_sets) == 1 and self.detector_sets[0] == "ALL":
            # Run the operators with all detectors at once
            for op in self.operators:
                msg = f"{pstr} {self} calling operator '{op.name}' exec() with ALL dets"
                log.verbose(msg)
                op.exec(data, detectors=None, use_accel=use_accel)
        elif len(self.detector_sets) == 1 and self.detector_sets[0] == "SINGLE":
            # Get superset of detectors across all observations
            all_local_dets = data.all_local_detectors(selection=detectors)
            # Run operators one detector at a time
            for det in all_local_dets:
                msg = f"{pstr} {self} SINGLE detector {det}"
                log.verbose(msg)
                for op in self.operators:
                    msg = f"{pstr} {self} calling operator '{op.name}' exec()"
                    log.verbose(msg)
                    op.exec(data, detectors=[det], use_accel=use_accel)
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
                msg = f"{pstr} {self} detector set {selected_set}"
                log.verbose(msg)
                for op in self.operators:
                    msg = f"{pstr} {self} calling operator '{op.name}' exec()"
                    log.verbose(msg)
                    op.exec(data, detectors=selected_set, use_accel=use_accel)
        return

    @function_timer
    def _stage_requirements_to_device(self, data):
        """Move required data to the device"""
        requires = self.requires()
        data.accel_create(requires)
        data.accel_update_device(requires)

    @function_timer
    def _finalize(self, data, use_accel=False, **kwargs):
        log = Logger.get()
        result = list()
        pstr = f"Proc ({data.comm.world_rank}, {data.comm.group_rank})"
        msg = f"{pstr} {self} finalize"
        log.verbose(msg)

        # FIXME:  We need to clarify in documentation that if using the
        # accelerator in _finalize() to produce output products, these
        # outputs should remain on the device so that they can be copied
        # out at the end automatically.

        if self.operators is not None:
            for op in self.operators:
                log.verbose(msg)
                result.append(op.finalize(data, use_accel=use_accel, **kwargs))

        # Copy out from accelerator if we did the copy in.
        if self._staged_accel:
            # Copy out the outputs to the CPU
            prov = self.provides()
            msg = f"{pstr} {self} copying out accel data outputs: {prov}"
            log.verbose(msg)
            data.accel_update_host(prov)
            # Delete the intermediate products from the GPU.  Otherwise, they will
            # get re-used by other pipelines despite still being on GPU.
            interm = self._get_intermediate()
            msg = f"{pstr} {self} deleting accel data intermediate outputs: {interm}"
            log.verbose(msg)
            data.accel_delete(interm)
            # Delete the inputs.  Otherwise, they will get re-used by other pipelines
            # despite still being on GPU.
            req = self.requires()
            msg = f"{pstr} {self} deleting accel data inputs: {req}"
            log.verbose(msg)
            data.accel_delete(req)
        return result

    def _requires(self):
        # Work through the operator list in reverse order and prune intermediate
        # products (that will be provided by a previous operator).
        if self.operators is None:
            return dict()
        keys = ["global", "meta", "detdata", "shared", "intervals"]
        req = {x: set() for x in keys}
        for op in reversed(self.operators):
            oreq = op.requires()
            oprov = op.provides()
            for k in keys:
                # remove provides first as there can be an overlap between provides and requires
                if k in oprov:
                    req[k] -= set(oprov[k])
                if k in oreq:
                    req[k] |= set(oreq[k])
        for k in keys:
            req[k] = list(req[k])

        return req

    def _provides(self):
        # Work through the operator list and prune intermediate products
        # (that are be provided to an intermediate operator).
        # FIXME could a final result also be used by an intermediate operator?
        if self.operators is None:
            return dict()
        keys = ["global", "meta", "detdata", "shared", "intervals"]
        prov = {x: set() for x in keys}
        for op in self.operators:
            oreq = op.requires()
            oprov = op.provides()
            for k in keys:
                # remove requires first as there can be an overlap between provides and requires
                if k in oreq:
                    prov[k] -= set(oreq[k])
                if k in oprov:
                    prov[k] |= set(oprov[k])
        for k in keys:
            prov[k] = list(prov[k])
        return prov

    def _get_intermediate(self):
        keys = ["global", "meta", "detdata", "shared", "intervals"]
        # Full provide minus intermediate
        prov = self.provides()
        interm = {x: set() for x in keys}
        for op in self.operators:
            oprov = op.provides()
            for k in keys:
                if k in oprov:
                    interm[k] |= set(oprov[k])
        # Deduce intermediate by subtraction
        for k in keys:
            interm[k] -= set(prov[k])
            interm[k] = list(interm[k])
        return interm

    def _implementations(self):
        """
        Find implementations supported by all the operators
        """
        implementations = {
            ImplementationType.DEFAULT,
            ImplementationType.COMPILED,
            ImplementationType.NUMPY,
            ImplementationType.JAX,
        }
        for op in self.operators:
            implementations.intersection_update(op.implementations())
        return list(implementations)

    def _supports_accel(self):
        """
        Returns True if all the operators are GPU compatible.
        """
        for op in self.operators:
            if not op.supports_accel():
                log = Logger.get()
                msg = f"{self} does not support accel because of '{op}'"
                log.debug(msg)
                return False
        return True

    def __str__(self):
        """
        Converts the pipeline into a human-readable string.
        """
        return f"Pipeline{[str(op) for op in self.operators]}"
