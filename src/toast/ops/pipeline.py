# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

from ..accelerator import ImplementationType, accel_enabled, use_hybrid_pipelines
from ..data import Data
from ..timing import function_timer
from ..traits import Bool, Int, List, trait_docs
from ..utils import Logger, SetDict
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

    use_hybrid = Bool(
        True,
        help="Should the pipeline be allowed to use the GPU when it has some cpu-only operators.",
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
        # keeps track of the data that is on device
        self._staged_data = None
        # keep track of the data that had to move back to host due to a cpu-only operator
        # (for display / debugging purposes)
        self._unstaged_data = None

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()
        pstr = f"Proc ({data.comm.world_rank}, {data.comm.group_rank})"

        if len(self.operators) == 0:
            log.debug_rank(
                "Pipeline has no operators, nothing to do", comm=data.comm.comm_world
            )
            return

        # If the calling code passed use_accel=True, we assume that it will move
        # the data for us.  Otherwise, if possible / allowed, use the accelerator and
        # deal with data movement ourselves.
        self._staged_data = None
        self._unstaged_data = None
        pipe_accel = self._pipe_accel(use_accel)

        if pipe_accel:
            # some of our operators support using the accelerator
            msg = f"{self} supports accelerators."
            log.verbose_rank(msg, comm=data.comm.comm_world)
            use_accel = True
            # keeps track of the data that is on device
            self._staged_data = SetDict(
                {
                    key: set()
                    for key in ["global", "meta", "detdata", "shared", "intervals"]
                }
            )
            # keep track of the data that had to move back from device
            # (for display / debugging purposes)
            self._unstaged_data = SetDict(
                {
                    key: set()
                    for key in ["global", "meta", "detdata", "shared", "intervals"]
                }
            )

        if len(data.obs) == 0:
            # No observations for this group
            msg = f"{self} data, group {data.comm.group} has no observations."
            log.verbose_rank(msg, comm=data.comm.comm_group)

        if len(self.detector_sets) == 1 and self.detector_sets[0] == "ALL":
            # Run the operators with all detectors at once
            for op in self.operators:
                self._exec_operator(
                    op,
                    data,
                    detectors=None,
                    pipe_accel=pipe_accel,
                )
        elif len(self.detector_sets) == 1 and self.detector_sets[0] == "SINGLE":
            # Get superset of detectors across all observations
            all_local_dets = data.all_local_detectors(selection=detectors)
            # Run operators one detector at a time
            for det in all_local_dets:
                msg = f"{pstr} {self} SINGLE detector {det}"
                log.verbose(msg)
                for op in self.operators:
                    self._exec_operator(
                        op,
                        data,
                        detectors=[det],
                        pipe_accel=pipe_accel,
                    )
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
                    self._exec_operator(
                        op,
                        data,
                        detectors=selected_set,
                        pipe_accel=pipe_accel,
                    )

        # notify user of device->host data movements introduced by CPU operators
        if (self._unstaged_data is not None) and (not self._unstaged_data.is_empty()):
            cpu_ops = {str(op) for op in self.operators if not op.supports_accel()}
            log.debug(
                f"{pstr} {self}, had to move {self._unstaged_data} back to host as {cpu_ops} do not support accel."
            )

    @function_timer
    def _exec_operator(self, op, data, detectors, pipe_accel):
        """Runs an operator, dealing with data movement to/from device if needed."""
        # For this operator, we run on the accelerator if the pipeline has some
        # operators enabled and if this operator supports it.
        run_accel = pipe_accel and op.supports_accel()

        log = Logger.get()
        msg = f"Proc ({data.comm.world_rank}, {data.comm.group_rank}) {self} "
        msg += f"calling operator '{op.name}' exec(accelerator={run_accel})"
        if detectors is None:
            msg += " with ALL dets"
        log.verbose(msg)

        # Ensures data is where it should be for this operator
        if self._staged_data is not None:
            requires = SetDict(op.requires())
            if run_accel:
                # This operator will use the accelerator, stage data
                msg = f"Proc ({data.comm.world_rank}, {data.comm.group_rank}) {self} "
                msg += f"BEFORE staged = {self._staged_data}, unstaged = {self._unstaged_data}"
                log.verbose(msg)
                requires -= self._staged_data
                msg = f"Proc ({data.comm.world_rank}, {data.comm.group_rank}) {self} "
                msg += f"Staging objects {requires}"
                log.verbose(msg)
                data.accel_create(requires)
                data.accel_update_device(requires)
                # Update our record of data on device
                self._unstaged_data -= requires
                self._staged_data |= requires
                self._staged_data |= op.provides()
                self._unstaged_data -= op.provides()
                msg = f"Proc ({data.comm.world_rank}, {data.comm.group_rank}) {self} "
                msg += f"AFTER staged = {self._staged_data}, unstaged = {self._unstaged_data}"
                log.verbose(msg)
            else:
                # This operator is running on the host, unstage data
                msg = f"Proc ({data.comm.world_rank}, {data.comm.group_rank}) {self} "
                msg += f"BEFORE staged = {self._staged_data}, unstaged = {self._unstaged_data}"
                log.verbose(msg)
                requires &= self._staged_data  # intersection
                msg = f"Proc ({data.comm.world_rank}, {data.comm.group_rank}) {self} "
                msg += f"Un-staging objects {requires}"
                log.verbose(msg)
                data.accel_update_host(requires)
                # Update our record of data on the device
                self._staged_data -= requires
                self._unstaged_data |= requires  # union
                self._unstaged_data |= op.provides()
                # lets operator decide if it wants to move data and operate on device by itself
                run_accel = None
                msg = f"Proc ({data.comm.world_rank}, {data.comm.group_rank}) {self} "
                msg += f"AFTER staged = {self._staged_data}, unstaged = {self._unstaged_data}"
                log.verbose(msg)
        # runs operator
        op.exec(data, detectors=detectors, use_accel=run_accel)

    @function_timer
    def _finalize(self, data, use_accel=None, **kwargs):
        # FIXME:  We need to clarify in documentation that if using the
        # accelerator in _finalize() to produce output products, these
        # outputs should remain on the device so that they can be copied
        # out at the end automatically.

        log = Logger.get()
        pstr = f"Proc ({data.comm.world_rank}, {data.comm.group_rank})"
        msg = f"{pstr} {self} finalize"
        log.verbose(msg)

        # Are we running on the accelerator?
        pipe_accel = self._pipe_accel(use_accel)

        # run finalize on all the operators in the pipeline
        # NOTE: this might produce some output products
        result = list()
        if self.operators is not None:
            for op in self.operators:
                # Did we set use_accel to true when running with this operator
                use_accel_op = pipe_accel and op.supports_accel()
                result.append(op.finalize(data, use_accel=use_accel_op, **kwargs))

        # get outputs back and clean up data
        # if we are in charge of the data movement
        if self._staged_data is not None:
            # get outputs back from device
            provides = SetDict(self.provides())
            provides &= self._staged_data  # intersection
            log.verbose(f"{pstr} {self} copying out accel data outputs: {provides}")
            data.accel_update_host(provides)
            # deleting all data on device
            log.verbose(f"{pstr} {self} deleting accel data: {self._staged_data}")
            data.accel_delete(self._staged_data)
            self._staged_data = None
            self._unstaged_data = None

        return result

    def _pipe_accel(self, use_accel):
        if (use_accel is None) and accel_enabled():
            # Only allows hybrid pipelines if the environement variable and pipeline agree to it
            # (they both default to True)
            use_hybrid = self.use_hybrid and use_hybrid_pipelines
            # can we run this pipelines on accelerator
            supports_accel = (
                self._supports_accel_partial() if use_hybrid else self._supports_accel()
            )
            return supports_accel
        else:
            return use_accel

    def _requires(self):
        """
        Work through the operator list in reverse order and prune intermediate products
        (that will be provided by a previous operator).
        """
        # constructs the union of the requires minus the provides (in reverse order)
        req = SetDict(
            {key: set() for key in ["global", "meta", "detdata", "shared", "intervals"]}
        )
        for op in reversed(self.operators):
            # remove provides first as there can be an overlap between provides and requires
            req -= op.provides()
            req |= op.requires()
        # converts into a dictionary of lists
        req = {k: list(v) for (k, v) in req.items()}
        return req

    def _provides(self):
        """
        Work through the operator list and prune intermediate products
        (that are be provided to an intermediate operator).
        FIXME could a final result also be used by an intermediate operator?
        """
        # constructs the union of the provides minus the requires
        prov = SetDict(
            {key: set() for key in ["global", "meta", "detdata", "shared", "intervals"]}
        )
        for op in self.operators:
            # remove requires first as there can be an overlap between provides and requires
            prov -= op.requires()
            prov |= op.provides()
        # converts into a dictionary of lists
        prov = {k: list(v) for (k, v) in prov.items()}
        return prov

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
        Returns True if *all* the operators are accelerator compatible.
        """
        for op in self.operators:
            if not op.supports_accel():
                return False
        return True

    def _supports_accel_partial(self):
        """
        Returns True if *at least one* of the operators is accelerator compatible.
        """
        for op in self.operators:
            if op.supports_accel():
                return True
        return False

    def __str__(self):
        """
        Converts the pipeline into a human-readable string.
        """
        return f"Pipeline{[str(op) for op in self.operators]}"
