# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import OrderedDict

import traitlets

from ..accelerator import accel_enabled
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
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        pstr = f"Proc ({data.comm.world_rank}, {data.comm.group_rank})"

        if len(self.operators) == 0:
            log.debug_rank(
                "Pipeline has no operators, nothing to do", comm=data.comm.comm_world
            )
            return

        # By default, if our own pipeline instance has use_accel set to True, then
        # it means that the calling code has already staged data to the device.
        self._staged_accel = False

        if not self.use_accel:
            # The calling code determined that we do not have all the data present to
            # use the accelerator.  HOWEVER, if our operators support it, we can stage
            # the data to and from the device.
            if detectors is None:
                comp_dets = set(data.all_local_detectors(selection=None))
            else:
                comp_dets = set(detectors)
            if accel_enabled() and self.supports_accel():
                # All our operators support it.
                msg = f"{self} fully supports accelerators, data to "
                msg += f"be staged: {self.requires()}"
                log.verbose_rank(msg, comm=data.comm.comm_world)
                self.use_accel = True
                # Save state of accel use in all operators, then enable them
                self._save_op_accel = list()
                for op in self.operators:
                    self._save_op_accel.append(op.use_accel)
                    op.use_accel = True

            # Deletes leftover intermediate values
            self._delete_intermediates(data, self.use_accel)

            # Send the requirements to the device
            self._stage_requirements_to_device(data, self.use_accel)

        if len(data.obs) == 0:
            # No observations for this group
            msg = f"{self} data, group {data.comm.group} has no observations."
            log.verbose_rank(msg, comm=data.comm.comm_group)

        if len(self.detector_sets) == 1 and self.detector_sets[0] == "ALL":
            # Run the operators with all detectors at once
            for op in self.operators:
                msg = f"{pstr} {self} calling operator '{op.name}' exec() with ALL dets"
                log.verbose(msg)
                op.exec(data, detectors=None)
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
                msg = f"{pstr} {self} detector set {selected_set}"
                log.verbose(msg)
                for op in self.operators:
                    msg = f"{pstr} {self} calling operator '{op.name}' exec()"
                    log.verbose(msg)
                    op.exec(data, detectors=selected_set)

        return

    @function_timer
    def _delete_intermediates(self, data, use_accel):
        """
        Deals with data existing on the host from a previous call, but
        not on the device.  Delete the host copy so that it will
        be re-created consistently.
        """
        if use_accel:
            log = Logger.get()
            interm = self._get_intermediate()
            log.verbose(f"intermediate = {interm}")
            for ob in data.obs:
                for obj in interm["detdata"]:
                    if obj in ob.detdata and not ob.detdata.accel_exists(obj):
                        msg = f"Pipeline intermediate detdata '{obj}' in "
                        msg += f"observation '{ob.name}' exists on "
                        msg += f"the host but not the device.  Deleting."
                        log.verbose_rank(msg, comm=data.comm.comm_group)
                        del ob.detdata[obj]
                for obj in interm["shared"]:
                    if obj in ob.shared and not ob.shared.accel_exists(obj):
                        msg = f"Pipeline intermediate shared data '{obj}' in "
                        msg += f"observation '{ob.name}' exists on "
                        msg += f"the host but not the device.  Deleting."
                        log.verbose_rank(msg, comm=data.comm.comm_group)
                        del ob.shared[obj]
                for obj in interm["intervals"]:
                    if obj in ob.intervals and not ob.intervals.accel_exists(obj):
                        msg = f"Pipeline intermediate intervals '{obj}' in "
                        msg += f"observation '{ob.name}' exists on "
                        msg += f"the host but not the device.  Deleting."
                        log.verbose_rank(msg, comm=data.comm.comm_group)
                        del ob.intervals[obj]

    @function_timer
    def _stage_requirements_to_device(self, data, use_accel):
        """move required data to the device"""
        if use_accel:
            requires = self.requires()
            data.accel_create(requires)
            data.accel_update_device(requires)
            self._staged_accel = True

    @function_timer
    def _finalize(self, data, **kwargs):
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
                result.append(op.finalize(data))

        # Copy out from accelerator if we did the copy in.
        if self._staged_accel:
            # Restore operator state
            for op, original in zip(self.operators, self._save_op_accel):
                op.use_accel = original

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
                if k in oreq:
                    req[k] |= set(oreq[k])
                if k in oprov:
                    req[k] -= set(oprov[k])
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
                if k in oprov:
                    prov[k] |= set(oprov[k])
                if k in oreq:
                    prov[k] -= set(oreq[k])
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
        # Find implementations supported by all the operators
        all_impl = [
            ImplementationType.DEFAULT,
            ImplementationType.COMPILED,
            ImplementationType.NUMPY,
            ImplementationType.JAX,
        ]
        impl = set(all_impl)
        for op in self.operators:
            for im in all_impl:
                if im not in op.implementations():
                    impl.remove(im)
        return list(impl)

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
