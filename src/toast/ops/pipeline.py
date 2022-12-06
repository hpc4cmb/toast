# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

from ..accelerator import accel_enabled
from ..data import Data
from ..timing import function_timer
from ..traits import Int, List, Unicode, trait_docs
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
    def _exec(self, data, detectors=None, use_accel=False, **kwargs):
        log = Logger.get()
        pstr = f"Proc ({data.comm.world_rank}, {data.comm.group_rank})"

        # If the calling code passed use_accel=True, we assume that it will move the data for us
        # otherwise, if possible, use the accelerator and deal with data movement ourselves
        self._staged_data = None
        if (not use_accel) and accel_enabled() and self._supports_accel_partial():
            # some of our operators support using the accelerator
            msg = f"{self} supports accelerators."
            log.verbose_rank(msg, comm=data.comm.comm_world)
            use_accel = True
            # keeps track of the data that we are moving to device
            self._staged_data = SetDict(
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
                self._exec_operator(op, data, detectors=None, use_accel=use_accel)
        elif len(self.detector_sets) == 1 and self.detector_sets[0] == "SINGLE":
            # Get superset of detectors across all observations
            all_local_dets = data.all_local_detectors(selection=detectors)
            # Run operators one detector at a time
            for det in all_local_dets:
                msg = f"{pstr} {self} SINGLE detector {det}"
                log.verbose(msg)
                for op in self.operators:
                    self._exec_operator(op, data, detectors=[det], use_accel=use_accel)
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
                        op, data, detectors=selected_set, use_accel=use_accel
                    )

    @function_timer
    def _exec_operator(self, op, data, detectors, use_accel):
        """Runs an operator, dealing with data movement to/from device if needed."""
        # displays some debugging information
        log = Logger.get()
        msg = f"Proc ({data.comm.world_rank}, {data.comm.group_rank}) {self} calling operator '{op.name}' exec()"
        if detectors is None:
            msg += " with ALL dets"
        log.verbose(msg)
        # insures data is where it should be for this operator
        if self._staged_data is not None:
            requires = SetDict(op.requires())
            if op.supports_accel():
                # get inputs not already on device
                requires -= self._staged_data
                data.accel_create(requires)
                data.accel_update_device(requires)
                # updates our record of data on device
                self._staged_data |= requires
                self._staged_data |= op.provides()
            else:
                # get inputs not already on host
                requires &= self._staged_data  # intersection
                data.accel_update_host(requires)
                # displays a message to push users to keep their operators device-able
                log = Logger.get()
                log.debug(
                    f"Had to move {requires} back to host as '{op}' does not support accel."
                )
                # updates our record of data on device
                self._staged_data -= requires
        # runs operator
        op.exec(data, detectors=detectors, use_accel=use_accel)

    @function_timer
    def _finalize(self, data, use_accel=False, **kwargs):
        log = Logger.get()
        pstr = f"Proc ({data.comm.world_rank}, {data.comm.group_rank})"
        msg = f"{pstr} {self} finalize"
        log.verbose(msg)

        # FIXME:  We need to clarify in documentation that if using the
        # accelerator in _finalize() to produce output products, these
        # outputs should remain on the device so that they can be copied
        # out at the end automatically.

        # did we set use_accel to True when running?
        use_accel = use_accel or (self._staged_data is not None)

        # run finalize on all the operators in the pipeline
        # NOTE: this might produce some output products
        result = list()
        if self.operators is not None:
            for op in self.operators:
                result.append(op.finalize(data, use_accel=use_accel))

        # get outputs back and clean up data
        # if we are in charge of the data movement
        if self._staged_data is not None:
            # get outputs back from device
            provides = SetDict(self.provides())
            provides &= self._staged_data  # intersection
            log.verbose(f"{pstr} {self} copying out accel data outputs: {provides}")
            data.accel_update_host(provides)
            # deleting all data from device
            log.verbose(f"{pstr} {self} deleting accel data: {self._staged_data}")
            data.accel_delete(self._staged_data)
            self._staged_data = None

        return result

    def _requires(self):
        """
        Work through the operator list in reverse order
        and prune intermediate products
        (that will be provided by a previous operator).
        """
        # constructs the union of the requires minus the provides (in reverse order)
        req = SetDict(
            {key: set() for key in ["global", "meta", "detdata", "shared", "intervals"]}
        )
        for op in reversed(self.operators):
            req |= op.requires()
            req -= op.provides()
        # converts into a dictionary of lists
        req = {k: list(v) for (k, v) in req.items()}
        return req

    def _provides(self):
        """
        Work through the operator list
        and prune intermediate products
        (that are be provided to an intermediate operator).
        FIXME could a final result also be used by an intermediate operator?
        """
        # constructs the union of the provides minus the requires
        prov = SetDict(
            {key: set() for key in ["global", "meta", "detdata", "shared", "intervals"]}
        )
        for op in self.operators:
            prov |= op.provides()
            prov -= op.requires()
        # converts into a dictionary of lists
        prov = {k: list(v) for (k, v) in prov.items()}
        return prov

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
        Returns True if at least one of the operators is accelerator compatible.
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
