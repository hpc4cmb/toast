# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets

from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Instance, Int, Unicode, trait_docs
from ..utils import Logger
from .delete import Delete
from .operator import Operator
from .pipeline import Pipeline


@trait_docs
class BuildPixelDistribution(Operator):
    """Operator which builds the pixel distribution information.

    This operator runs the pointing operator and builds the PixelDist instance
    describing how submaps are distributed among processes.  This requires expanding
    the full detector pointing once in order to compute the distribution.  This is
    done one detector at a time unless the save_pointing trait is set to True.

    NOTE:  The pointing operator must have the "pixels" and "create_dist"
    traits, which will be set by this operator during execution.

    Output PixelDistribution objects are stored in the Data dictionary.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    pixel_dist = Unicode(
        "pixel_dist",
        help="The Data key where the PixelDist object should be stored",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    pixel_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pointing operator",
    )

    save_pointing = Bool(
        False, help="If True, do not clear detector pointing matrices after use"
    )

    @traitlets.validate("pixel_pointing")
    def _check_pixel_pointing(self, proposal):
        pntg = proposal["value"]
        if pntg is not None:
            if not isinstance(pntg, Operator):
                raise traitlets.TraitError(
                    "pixel_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["pixels", "create_dist", "view"]:
                if not pntg.has_trait(trt):
                    msg = f"pixel_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return pntg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in ("pixel_pointing",):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        if self.pixel_dist in data:
            msg = f"pixel distribution `{self.pixel_dist}` already exists"
            raise RuntimeError(msg)

        if detectors is not None:
            msg = "A subset of detectors is specified, but the pixel distribution\n"
            msg += "does not yet exist- and creating this requires all detectors."
            raise RuntimeError(msg)

        msg = "Creating pixel distribution '{}' in Data".format(self.pixel_dist)
        if data.comm.world_rank == 0:
            log.debug(msg)

        # Turn on creation of the pixel distribution
        self.pixel_pointing.create_dist = self.pixel_dist

        # Compute the pointing matrix

        pixel_dist_pipe = None
        if self.save_pointing:
            # We are keeping the pointing, which means we need to run all detectors
            # at once so they all end up in the detdata for all observations.
            pixel_dist_pipe = Pipeline(detector_sets=["ALL"])
        else:
            # Run one detector a at time and discard.
            pixel_dist_pipe = Pipeline(detector_sets=["SINGLE"])
        pixel_dist_pipe.operators = [
            self.pixel_pointing,
        ]
        # FIXME: Disable accelerator use for now, since it is a small amount of
        # calculation for a huge data volume.
        pipe_out = pixel_dist_pipe.apply(data, detectors=detectors, use_accel=False)

        # Turn pixel distribution creation off again
        self.pixel_pointing.create_dist = None

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.pixel_pointing.requires()
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        return req

    def _provides(self):
        prov = {
            "global": [self.pixel_dist],
            "shared": list(),
            "detdata": list(),
        }
        if self.save_pointing:
            prov["detdata"].extend([self.pixels])
        return prov
