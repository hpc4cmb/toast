# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ..covariance import covariance_apply
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Unicode, Unit, trait_docs
from ..utils import Logger
from .delete import Delete
from .accum_obs import AccumulateObservation
from .operator import Operator
from .pipeline import Pipeline


@trait_docs
class BinMap(Operator):
    """Operator which bins a map.

    Given a noise model and a pointing model, build the noise weighted map and
    apply the noise covariance to get resulting binned map.

    If the covariance does not yet exist, it is accumulated, optionally with caching.

    By default, detector pointing is expanded for a whole observation at a time and
    then deleted.  If `full_pointing` is True, all the pointing is expanded and saved.
    If `single_det_pointing` is True, the pointing is expanded one detector at a time
    for each observation.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    pixel_dist = Unicode(
        "pixel_dist",
        help="The Data key where the PixelDist object is stored",
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

    hits = Unicode(
        None,
        allow_none=True,
        help="Also accumulate the hit map and store in this Data key",
    )

    inverse_covariance = Unicode(
        None,
        allow_none=True,
        help="The Data key where the inverse covariance should be stored",
    )

    rcond = Unicode(
        None,
        allow_none=True,
        help="The Data key for the reciprocal condition number, if computed",
    )

    rcond_threshold = Float(
        1.0e-8, help="Minimum value for inverse condition number cut."
    )

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    det_data_units = Unit(defaults.det_data_units, help="Desired timestream units")

    det_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for per-detector flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for detector sample flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_nonscience,
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

    post_process = Instance(
        klass=Operator,
        allow_none=True,
        help="Optional extra operator to run after accumulation",
    )

    noise_model = Unicode(
        defaults.noise_model, help="Observation key containing the noise model"
    )

    sync_type = Unicode(
        "alltoallv", help="Communication algorithm: 'allreduce' or 'alltoallv'"
    )

    full_pointing = Bool(
        False, help="If True, save pointing for all observations and detectors"
    )

    single_det_pointing = Bool(
        False, help="If True, expand pointing one detector at a time"
    )

    cache_dir = Unicode(
        None,
        allow_none=True,
        help="Directory of per-observation cache directories for reading / writing",
    )

    overwrite_cache = Bool(
        False,
        help="If True and using a cache, overwrite any inputs found there",
    )

    cache_only = Bool(
        False,
        help="If True, do not accumulate the total products. Useful for pre-caching",
    )

    cache_detdata = Bool(
        False,
        help="If True, also cache the detector data",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
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
        timer = Timer()
        timer.start()

        for trait in "pixel_pointing", "stokes_weights", "det_data":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        log.verbose_rank("  BinMap building pipeline", comm=data.comm.comm_world)

        if self.covariance not in data:
            cov = None
            print(f"BinMap covariance {self.covariance} does not exist, will build", flush=True)
        else:
            cov = data[self.covariance]
            print(f"BinMap covariance {self.covariance} already exists", flush=True)

            # Check that covariance has consistent units
            if cov.units != (self.det_data_units**2).decompose():
                msg = f"Covariance '{self.covariance}' units {cov.units} do not"
                msg += f" equal det_data units ({self.det_data_units}) squared."
                log.error(msg)
                raise RuntimeError(msg)

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

        if self.binned in data:
            indata = data[self.binned]
            nonz = indata.data != 0
            print(f"BinMap input zmap = {indata.data[nonz]}", flush=True)

        # Use the same detector mask in the pointing
        self.pixel_pointing.detector_pointing.det_mask = self.det_mask
        self.pixel_pointing.detector_pointing.det_flag_mask = self.det_flag_mask
        if hasattr(self.stokes_weights, "detector_pointing"):
            self.stokes_weights.detector_pointing.det_mask = self.det_mask
            self.stokes_weights.detector_pointing.det_flag_mask = self.det_flag_mask

        # Set up the Accumulation operator
        if cov is None:
            accum_cov = self.covariance
            accum_invcov = self.inverse_covariance
            if accum_invcov is None:
                accum_invcov = f"{self.name}_invcov"
            accum_rcond = self.rcond
            if accum_rcond is None:
                accum_rcond = f"{self.name}_rcond"
        else:
            accum_cov = None
            accum_invcov = None
            accum_rcond = None
        obs_accum = AccumulateObservation(
            cache_dir=self.cache_dir,
            overwrite_cache=self.overwrite_cache,
            cache_only=self.cache_only,
            cache_detdata=self.cache_detdata,
            pixel_dist=self.pixel_dist,
            inverse_covariance=accum_invcov,
            hits=self.hits,
            rcond=accum_rcond,
            zmap=self.binned,
            covariance=accum_cov,
            det_data=self.det_data,
            det_mask=self.det_mask,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            det_data_units=self.det_data_units,
            shared_flags=self.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            pixel_pointing=self.pixel_pointing,
            stokes_weights=self.stokes_weights,
            obs_pointing=(not self.single_det_pointing),
            save_pointing=self.full_pointing,
            noise_model=self.noise_model,
            rcond_threshold=self.rcond_threshold,
            sync_type=self.sync_type,
        )
        print(f"BIN accum cache_detdata: {self.cache_detdata} -> {obs_accum.cache_detdata}", flush=True)

        # Build a pipeline to expand pointing and accumulate

        accum = None
        accum_ops = list()
        if self.pre_process is not None:
            accum_ops.append(self.pre_process)
        accum_ops.append(obs_accum)
        if self.post_process is not None:
            accum_ops.append(self.post_process)

        if self.single_det_pointing:
            # Process one detector at a time.
            accum = Pipeline(detector_sets=["SINGLE"])
        else:
            # Process all detectors in an observation at once, optionally
            # saving the pointing through options to AccumulateObservation
            # above.
            accum = Pipeline(detector_sets=["ALL"])
        accum.operators = accum_ops

        if data.comm.world_rank == 0:
            log.verbose("  BinMap running pipeline")

        # Use `load_apply()` to run the pipeline on one observation at a time,
        # optionally loading data on the fly if a loader is defined in each
        # observation.
        pipe_out = accum.load_apply(data, detectors=detectors)

        # If we built the covariance, and are not keeping the intermediate products,
        # delete them now.
        if cov is None:
            if self.inverse_covariance is None:
                del data[accum_invcov]
            if self.rcond is None:
                del data[accum_rcond]

        # Optionally, store the noise-weighted map before applying the covariance
        # in place.
        if self.noiseweighted is not None:
            data[self.noiseweighted] = data[self.binned].duplicate()

        if cov is None:
            # We computed it in our pipeline
            cov = data[self.covariance]

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
