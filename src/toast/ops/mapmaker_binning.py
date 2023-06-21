# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ..covariance import covariance_apply
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Instance, Int, Unicode, Unit, trait_docs
from ..utils import Logger
from .delete import Delete
from .mapmaker_utils import BuildHitMap, BuildInverseCovariance, BuildNoiseWeighted
from .operator import Operator
from .pipeline import Pipeline


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

    noiseweighted = Unicode(
        None,
        allow_none=True,
        help="The Data key where the noiseweighted map should be stored",
    )

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    det_data_units = Unit(defaults.det_data_units, help="Desired timestream units")

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for optional telescope flagging",
    )

    pixel_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pixel pointing operator",
    )

    stokes_weights = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a Stokes weights operator",
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
        "alltoallv", help="Communication algorithm: 'allreduce' or 'alltoallv'"
    )

    full_pointing = Bool(
        False, help="If True, expand pointing for all detectors and save"
    )

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

    @traitlets.validate("pixel_pointing")
    def _check_pixel_pointing(self, proposal):
        pixels = proposal["value"]
        if pixels is not None:
            if not isinstance(pixels, Operator):
                raise traitlets.TraitError(
                    "pixel_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["pixels", "create_dist", "view"]:
                if not pixels.has_trait(trt):
                    msg = f"pixel_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return pixels

    @traitlets.validate("stokes_weights")
    def _check_stokes_weights(self, proposal):
        weights = proposal["value"]
        if weights is not None:
            if not isinstance(weights, Operator):
                raise traitlets.TraitError(
                    "stokes_weights should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["weights", "view"]:
                if not weights.has_trait(trt):
                    msg = f"stokes_weights operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return weights

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "pixel_pointing", "stokes_weights":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        if data.comm.world_rank == 0:
            log.verbose("  BinMap building pipeline")

        if self.covariance not in data:
            msg = f"Data does not contain noise covariance '{self.covariance}'"
            log.error(msg)
            raise RuntimeError(msg)

        cov = data[self.covariance]

        # Check that covariance has consistent units
        if cov.units != (self.det_data_units**2).decompose():
            msg = f"Covariance '{self.covariance}' units {cov.units} do not"
            msg += f" equal det_data units ({self.det_data_units}) squared."
            log.error(msg)
            raise RuntimeError(msg)

        # Check that the detector data is set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")

        # Sanity check that the covariance pixel distribution agrees
        if cov.distribution != data[self.pixel_dist]:
            msg = (
                f"Pixel distribution '{self.pixel_dist}' does not match the one "
                f"used by covariance '{self.covariance}'"
            )
            log.error(msg)
            raise RuntimeError(msg)

        # Set outputs of the pointing operator

        self.pixel_pointing.create_dist = None

        # If the binned map already exists in the data, verify the distribution and
        # reset to zero.

        if self.binned in data:
            if data[self.binned].distribution != data[self.pixel_dist]:
                msg = (
                    f"Pixel distribution '{self.pixel_dist}' does not match "
                    f"existing binned map '{self.binned}'"
                )
                log.error(msg)
                raise RuntimeError(msg)
            data[self.binned].reset()
            data[self.binned].update_units(1.0 / self.det_data_units)

        # Noise weighted map.  We output this to the final binned map location,
        # since we will multiply by the covariance in-place.

        build_zmap = BuildNoiseWeighted(
            pixel_dist=self.pixel_dist,
            zmap=self.binned,
            view=self.pixel_pointing.view,
            pixels=self.pixel_pointing.pixels,
            weights=self.stokes_weights.weights,
            noise_model=self.noise_model,
            det_data=self.det_data,
            det_data_units=self.det_data_units,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            shared_flags=self.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            sync_type=self.sync_type,
        )

        # Build a pipeline to expand pointing and accumulate

        accum = None
        accum_ops = list()
        if self.pre_process is not None:
            accum_ops.append(self.pre_process)
        if self.full_pointing:
            # Process all detectors at once
            accum = Pipeline(detector_sets=["ALL"])
        else:
            # Process one detector at a time.
            accum = Pipeline(detector_sets=["SINGLE"])
        accum_ops.extend([self.pixel_pointing, self.stokes_weights, build_zmap])

        accum.operators = accum_ops

        if data.comm.world_rank == 0:
            log.verbose("  BinMap running pipeline")
        pipe_out = accum.apply(data, detectors=detectors)

        # print("Binned zmap = ", data[self.binned].data)

        # Optionally, store the noise-weighted map
        if self.noiseweighted is not None:
            data[self.noiseweighted] = data[self.binned].duplicate()

        # Extract the results
        binned_map = data[self.binned]

        # Apply the covariance in place
        if data.comm.world_rank == 0:
            log.verbose("  BinMap applying covariance")
        covariance_apply(cov, binned_map, use_alltoallv=(self.sync_type == "alltoallv"))
        # print("Binned final = ", data[self.binned].data)
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.pixel_pointing.requires()
        req.update(self.stokes_weights.requires())
        if self.pre_process is not None:
            req.update(self.pre_process.requires())
        req["global"].extend([self.pixel_dist, self.covariance])
        req["meta"].extend([self.noise_model])
        req["detdata"].extend([self.det_data])
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.pre_process is not None:
            req.update(self.pre_process.requires())
        return req

    def _provides(self):
        prov = {"global": [self.binned]}
        return prov
