# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance

from ..timing import function_timer

from ..pixels import PixelDistribution, PixelData

from ..covariance import covariance_apply

from .operator import Operator

from .pipeline import Pipeline

from .delete import Delete

from .mapmaker_utils import BuildHitMap, BuildNoiseWeighted, BuildInverseCovariance


@trait_docs
class BinMap(Operator):
    """Operator which bins a map.

    Given a noise model and a pointing operator, build the noise weighted map and
    apply the noise covariance to get resulting binned map.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    pixel_dist = Unicode(
        "pixel_dist",
        help="The Data key where the PixelDist object should be stored",
    )

    covariance = Unicode(
        "covariance",
        help="The Data key containing the noise covariance PixelData instance",
    )

    binned = Unicode(
        "binned",
        help="The Data key where the binned map should be stored",
    )

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional telescope flagging")

    pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pointing operator",
    )

    pre_process = Instance(
        klass=Operator,
        allow_none=True,
        help="Optional extra operator to run prior to binning",
    )

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    sync_type = Unicode(
        "allreduce", help="Communication algorithm: 'allreduce' or 'alltoallv'"
    )

    saved_pointing = Bool(False, help="If True, use previously computed pointing")

    @traitlets.validate("det_flag_mask")
    def _check_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("sync_type")
    def _check_sync_type(self, proposal):
        check = proposal["value"]
        if check != "allreduce" and check != "alltoallv":
            raise traitlets.TraitError("Invalid communication algorithm")
        return check

    @traitlets.validate("pointing")
    def _check_pointing(self, proposal):
        pntg = proposal["value"]
        if pntg is not None:
            if not isinstance(pntg, Operator):
                raise traitlets.TraitError("pointing should be an Operator instance")
            # Check that this operator has the traits we expect
            for trt in ["pixels", "weights", "create_dist", "view"]:
                if not pntg.has_trait(trt):
                    msg = "pointing operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return pntg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        data.comm.comm_world.barrier()
        if data.comm.world_rank == 0:
            log.verbose("  BinMap building pipeline")

        if self.covariance not in data:
            msg = "Data does not contain noise covariance '{}'".format(self.covariance)
            log.error(msg)
            raise RuntimeError(msg)

        cov = data[self.covariance]

        # Check that the detector data is set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")

        # Sanity check that the covariance pixel distribution agrees
        if cov.distribution != data[self.pixel_dist]:
            msg = "Pixel distribution '{}' does not match the one used by covariance '{}'".format(
                self.pixel_dist, self.covariance
            )
            log.error(msg)
            raise RuntimeError(msg)

        # Set outputs of the pointing operator

        self.pointing.create_dist = None

        # If the binned map already exists in the data, verify the distribution and
        # reset to zero.

        if self.binned in data:
            if data[self.binned].distribution != data[self.pixel_dist]:
                msg = "Pixel distribution '{}' does not match existing binned map '{}'".format(
                    self.pixel_dist, self.binned
                )
                log.error(msg)
                raise RuntimeError(msg)
            data[self.binned].raw[:] = 0.0

        # Noise weighted map.  We output this to the final binned map location,
        # since we will multiply by the covariance in-place.

        build_zmap = BuildNoiseWeighted(
            pixel_dist=self.pixel_dist,
            zmap=self.binned,
            view=self.pointing.view,
            pixels=self.pointing.pixels,
            weights=self.pointing.weights,
            noise_model=self.noise_model,
            det_data=self.det_data,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            sync_type=self.sync_type,
        )

        # Build a pipeline to expand pointing and accumulate

        accum = None
        accum_ops = list()
        if self.pre_process is not None:
            accum_ops.append(self.pre_process)
        if self.saved_pointing:
            # Process all detectors at once, using existing pointing
            accum = Pipeline(detector_sets=["ALL"])
            accum_ops.extend([build_zmap])
        else:
            # Process one detector at a time.
            accum = Pipeline(detector_sets=["SINGLE"])
            accum_ops.extend([self.pointing, build_zmap])

        accum.operators = accum_ops

        if data.comm.world_rank == 0:
            log.verbose("  BinMap running pipeline")
        pipe_out = accum.apply(data, detectors=detectors)

        # Extract the results
        binned_map = data[self.binned]

        # Apply the covariance in place
        if data.comm.world_rank == 0:
            log.verbose("  BinMap applying covariance")
        covariance_apply(cov, binned_map, use_alltoallv=(self.sync_type == "alltoallv"))

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.pointing.requires()
        req["meta"].extend([self.noise_model, self.pixel_dist, self.covariance])
        req["detdata"].extend([self.det_data])
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {"meta": [self.binned], "shared": list(), "detdata": list()}
        return prov

    def _accelerators(self):
        return list()
